/// @file layernorm.cu
/// @brief Fused RMSNorm — CUDA kernel.
///
/// Implements RMSNorm as used in Llama-2:
///   y = x * rsqrt(mean(x^2) + eps) * weight
///
/// This is NOT LayerNorm (no mean subtraction). RMSNorm is simpler
/// and faster, making it the standard for modern LLMs.


#include "cuda/kernels.h"
#include "cuda/utils.h"
#include <cuda_runtime.h>

namespace vulcan {
namespace cuda {

// ─── Constants ──────────────────────────────────────────────────────────────

constexpr int NORM_BLOCK_SIZE = 256;

// ─── CUDA Kernel ────────────────────────────────────────────────────────────

/// RMSNorm kernel.
///
/// Algorithm (per vector of dimension n):
///   1. Compute sum of squares: ss = sum(x[i]^2) for i in [0, n)
///   2. Compute RMS: rms = rsqrt(ss / n + eps)
///   3. Apply: y[i] = x[i] * rms * weight[i]
///
/// Optimization notes:
///   - Use warp-level reduction (__shfl_down_sync) for the sum
///   - Use shared memory for cross-warp reduction
///   - Can be fused with subsequent MatMul for bandwidth savings
///
/// @param input   Input vector [n] (device)
/// @param weight  Scale parameters [n] (device)
/// @param output  Output vector [n] (device)
/// @param n       Vector dimension
/// @param eps     Numerical stability epsilon
__global__ void rmsnorm_kernel(const float* __restrict__ input,
                               const float* __restrict__ weight,
                               float* __restrict__ output,
                               int n, float eps) {
    __shared__ float shared_sum[NORM_BLOCK_SIZE / 32]; // One per warp

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // Step 1: Each thread accumulates partial sum of squares
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float val = input[i];
        local_sum += val * val;
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }

    // Write warp result to shared memory
    if (lane_id == 0) {
        shared_sum[warp_id] = local_sum;
    }
    __syncthreads();

    // First warp reduces across all warps
    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x / 32)) ? shared_sum[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if (lane_id == 0) {
            shared_sum[0] = val;
        }
    }
    __syncthreads();

    // Step 2: Compute rsqrt(mean(x^2) + eps)
    float rms = rsqrtf(shared_sum[0] / static_cast<float>(n) + eps);

    // Step 3: Apply normalization and scale
    for (int i = tid; i < n; i += blockDim.x) {
        output[i] = input[i] * rms * weight[i];
    }
}

// ─── Host Wrapper ───────────────────────────────────────────────────────────

void launch_rmsnorm(const float* input, const float* weight, float* output,
                    int n, float eps) {
    // Single block launch — one normalization at a time
    // For batched norm, need to extend grid to cover batch dimension
    int block_size = (n < NORM_BLOCK_SIZE) ? n : NORM_BLOCK_SIZE;
    // Round up to nearest warp
    block_size = ((block_size + 31) / 32) * 32;
    if (block_size > NORM_BLOCK_SIZE) block_size = NORM_BLOCK_SIZE;

    rmsnorm_kernel<<<1, block_size>>>(input, weight, output, n, eps);

    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();
}

} // namespace cuda
} // namespace vulcan
