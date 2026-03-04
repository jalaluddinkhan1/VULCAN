/// @file fused_kernels.cu
/// @brief Fused CUDA kernels for reduced memory bandwidth.
///
/// Kernel fusion eliminates intermediate global memory reads/writes
/// between adjacent operations. Per ADR-002, we fuse:
///
///   1. RMSNorm + Linear Projection (saves one global memory round-trip)
///   2. SiLU(gate) * up (3 memory ops → 2 reads + 1 write)
///   3. Dequantize + MatMul (INT4 weights dequantized in-register)


#include "cuda/kernels.h"
#include "cuda/utils.h"
#include <cuda_runtime.h>
#include <cfloat>

namespace vulcan {
namespace cuda {

// ═══════════════════════════════════════════════════════════════════════════
//  Fused SiLU(gate) * up — Gated MLP Activation
// ═══════════════════════════════════════════════════════════════════════════

/// Fuses silu(gate[i]) * up[i] into a single kernel.
/// Eliminates the intermediate SiLU output buffer entirely.
///
/// Before fusion (3 kernel launches, 1 temp buffer):
///   temp = silu(gate)       ← 1 read + 1 write
///   out  = temp * up        ← 2 reads + 1 write
///
/// After fusion (1 kernel, 0 temp buffers):
///   out = silu(gate[i]) * up[i]  ← 2 reads + 1 write
__global__ void fused_silu_mul_kernel(const float* __restrict__ gate,
                                      const float* __restrict__ up,
                                      float* __restrict__ output,
                                      int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = gate[idx];
        float silu_g = g / (1.0f + expf(-g));  // SiLU
        output[idx] = silu_g * up[idx];          // Gate * Up
    }
}

void launch_fused_silu_mul(const float* gate, const float* up,
                           float* output, int n) {
    const int block_size = 256;
    const int grid_size = calc_grid_1d(n, block_size);
    fused_silu_mul_kernel<<<grid_size, block_size>>>(gate, up, output, n);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();
}

// ═══════════════════════════════════════════════════════════════════════════
//  Fused RMSNorm + Linear Projection
// ═══════════════════════════════════════════════════════════════════════════

/// Fuses RMSNorm(x) @ W into a single kernel.
///
/// Before fusion:
///   normed = rmsnorm(x, weight)     ← read x, write normed
///   output = normed @ W             ← read normed, read W, write output
///
/// After fusion:
///   RMSNorm computed in shared memory, immediately used for dot product.
///   Eliminates the intermediate `normed` buffer.
///
/// Each thread block processes one row (one token position).
/// Within the block, threads cooperate to:
///   1. Compute sum of squares for RMSNorm (parallel reduction)
///   2. Compute normalized values in shared memory
///   3. Each output column computed via dot product with weight column
///
/// @param input       [seq_len, dim] input hidden states
/// @param norm_weight [dim] RMSNorm scale parameters
/// @param proj_weight [dim, out_dim] linear projection weights (row-major)
/// @param output      [seq_len, out_dim] output
/// @param seq_len     Number of tokens
/// @param dim         Input dimension
/// @param out_dim     Output dimension
/// @param eps         RMSNorm epsilon
__global__ void fused_rmsnorm_linear_kernel(
    const float* __restrict__ input,
    const float* __restrict__ norm_weight,
    const float* __restrict__ proj_weight,
    float* __restrict__ output,
    int seq_len, int dim, int out_dim, float eps) {

    extern __shared__ float shared[];
    // shared[0..dim-1] = normalized input for this row
    // shared[dim]      = RMS value

    int row = blockIdx.x;      // Token position
    int tid = threadIdx.x;

    if (row >= seq_len) return;

    const float* x = input + row * dim;

    // ── Step 1: Compute sum of squares ──────────────────────────────
    float local_sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float val = x[i];
        local_sum += val * val;
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }

    // Cross-warp reduction via shared memory
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = blockDim.x / 32;

    if (lane_id == 0) {
        shared[dim + warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? shared[dim + lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if (lane_id == 0) {
            shared[dim] = rsqrtf(val / static_cast<float>(dim) + eps);
        }
    }
    __syncthreads();

    float rms = shared[dim];

    // ── Step 2: Compute normalized values into shared memory ────────
    for (int i = tid; i < dim; i += blockDim.x) {
        shared[i] = x[i] * rms * norm_weight[i];
    }
    __syncthreads();

    // ── Step 3: Linear projection — one dot product per output col ──
    float* out_row = output + row * out_dim;
    for (int col = tid; col < out_dim; col += blockDim.x) {
        float dot = 0.0f;
        for (int k = 0; k < dim; ++k) {
            dot += shared[k] * proj_weight[k * out_dim + col];
        }
        out_row[col] = dot;
    }
}

void launch_fused_rmsnorm_linear(
    const float* input, const float* norm_weight,
    const float* proj_weight, float* output,
    int seq_len, int dim, int out_dim, float eps) {

    int block_size = 256;
    // Shared memory: dim floats (normalized) + max_warps floats (reduction)
    size_t shared_mem = (dim + block_size / 32 + 1) * sizeof(float);

    fused_rmsnorm_linear_kernel<<<seq_len, block_size, shared_mem>>>(
        input, norm_weight, proj_weight, output,
        seq_len, dim, out_dim, eps
    );
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();
}

// ═══════════════════════════════════════════════════════════════════════════
//  Quantized MatMul — INT4 weights dequantized in-register
// ═══════════════════════════════════════════════════════════════════════════

/// Quantized matrix multiply: C = A * dequant(B_int4)
///
/// B is stored as packed INT4 with per-group scale factors.
/// Each thread computes one element of C by:
///   1. Loading A row values from global memory
///   2. Loading packed INT4 bytes from B
///   3. Dequantizing in-register (unpack nibble, subtract 8, multiply scale)
///   4. Accumulating dot product via FMA
///
/// NEVER writes dequantized weights to global memory — saves 8× bandwidth.
///
/// @param A           [M, K] float input (device)
/// @param B_packed    [K/2, N] packed INT4 weights (device)
/// @param scales      [K/group_size, N] per-group scale factors (device)
/// @param C           [M, N] float output (device)
/// @param M           Rows of A (seq_len or 1 for decode)
/// @param K           Inner dimension
/// @param N           Columns of output
/// @param group_size  Quantization group size
__global__ void quantized_matmul_kernel(
    const float* __restrict__ A,
    const uint8_t* __restrict__ B_packed,
    const float* __restrict__ scales,
    float* __restrict__ C,
    int M, int K, int N, int group_size) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;

    // Process K dimension, 2 elements at a time (one packed byte)
    for (int k = 0; k < K; k += 2) {
        // Load packed byte: contains B[k, col] and B[k+1, col]
        int byte_row = k / 2;
        uint8_t packed = B_packed[byte_row * N + col];

        // Unpack and dequantize
        int val_low  = static_cast<int>(packed & 0x0F) - 8;
        int val_high = static_cast<int>(packed >> 4) - 8;

        int group_low  = k / group_size;
        int group_high = (k + 1) / group_size;

        float scale_low  = scales[group_low * N + col];
        float scale_high = scales[group_high * N + col];

        float b_low  = static_cast<float>(val_low)  * scale_low;
        float b_high = static_cast<float>(val_high) * scale_high;

        // FMA accumulation
        sum += A[row * K + k]     * b_low;
        sum += A[row * K + k + 1] * b_high;
    }

    C[row * N + col] = sum;
}

void launch_quantized_matmul(
    const float* A, const uint8_t* B_packed, const float* scales,
    float* C, int M, int K, int N, int group_size) {

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    quantized_matmul_kernel<<<grid, block>>>(
        A, B_packed, scales, C, M, K, N, group_size
    );
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();
}

// ═══════════════════════════════════════════════════════════════════════════
//  Fused Residual + RMSNorm
// ═══════════════════════════════════════════════════════════════════════════

/// Fuses residual add + RMSNorm into a single kernel.
/// output = rmsnorm(residual + hidden, weight, eps)
///
/// Eliminates the intermediate post-residual buffer.
__global__ void fused_residual_rmsnorm_kernel(
    const float* __restrict__ residual,
    const float* __restrict__ hidden,
    const float* __restrict__ weight,
    float* __restrict__ output,
    float* __restrict__ residual_out,  // optionally save residual sum
    int n, float eps) {

    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // Step 1: Residual add + accumulate sum of squares
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float val = residual[i] + hidden[i];
        if (residual_out) residual_out[i] = val;  // Save for next residual
        sdata[i] = val;  // Temp storage in shared (reuse for norm output)
        local_sum += val * val;
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }

    int num_warps_val = blockDim.x / 32;
    float* warp_sums = sdata + n;  // After the input data

    if (lane_id == 0) {
        warp_sums[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < num_warps_val) ? warp_sums[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if (lane_id == 0) {
            warp_sums[0] = rsqrtf(val / static_cast<float>(n) + eps);
        }
    }
    __syncthreads();

    float rms = warp_sums[0];

    // Step 2: Apply normalization
    for (int i = tid; i < n; i += blockDim.x) {
        // Note: sdata[i] contains (residual + hidden) from step 1
        output[i] = sdata[i] * rms * weight[i];
    }
}

void launch_fused_residual_rmsnorm(
    const float* residual, const float* hidden, const float* weight,
    float* output, float* residual_out, int n, float eps) {

    int block_size = 256;
    if (n < block_size) {
        block_size = ((n + 31) / 32) * 32;
        if (block_size > 256) block_size = 256;
    }

    // Shared: n floats (residual sum) + max_warps floats (reduction scratch)
    size_t shared_mem = (n + block_size / 32 + 1) * sizeof(float);

    fused_residual_rmsnorm_kernel<<<1, block_size, shared_mem>>>(
        residual, hidden, weight, output, residual_out, n, eps
    );
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();
}

} // namespace cuda
} // namespace vulcan
