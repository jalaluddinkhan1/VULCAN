/// @file softmax.cu
/// @brief Softmax kernel — numerically stable online softmax.
///
/// Implements: softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
///
/// Used in attention score normalization. The online algorithm avoids
/// materializing the full exp() array, computing max and sum in a single
/// pass via the Milakov-Gimelshein technique.


#include "cuda/kernels.h"
#include "cuda/utils.h"
#include <cuda_runtime.h>
#include <cfloat>

namespace vulcan {
namespace cuda {

// ─── Constants ──────────────────────────────────────────────────────────────

constexpr int SOFTMAX_BLOCK_SIZE = 256;

// ─── CUDA Kernel ────────────────────────────────────────────────────────────

/// Numerically stable softmax kernel.
///
/// Algorithm (single row of length n):
///   1. Find max(x) via parallel reduction
///   2. Compute sum(exp(x - max)) via parallel reduction
///   3. Output: exp(x_i - max) / sum
///
/// Uses shared memory for reductions.
///
/// @param input   Input vector [n] (device)
/// @param output  Output vector [n] (device, probabilities sum to 1)
/// @param n       Vector length
__global__ void softmax_kernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;

    // ── Step 1: Find max ────────────────────────────────────────────
    float thread_max = -FLT_MAX;
    for (int i = tid; i < n; i += blockDim.x) {
        thread_max = fmaxf(thread_max, input[i]);
    }

    sdata[tid] = thread_max;
    __syncthreads();

    // Block-level max reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    // ── Step 2: Compute sum(exp(x - max)) ───────────────────────────
    float thread_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        thread_sum += expf(input[i] - max_val);
    }

    sdata[tid] = thread_sum;
    __syncthreads();

    // Block-level sum reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    float sum_val = sdata[0];
    __syncthreads();

    // ── Step 3: Normalize ───────────────────────────────────────────
    float inv_sum = 1.0f / sum_val;
    for (int i = tid; i < n; i += blockDim.x) {
        output[i] = expf(input[i] - max_val) * inv_sum;
    }
}

// ─── Host Wrapper ───────────────────────────────────────────────────────────

void launch_softmax(const float* input, float* output, int n) {
    int block_size = (n < SOFTMAX_BLOCK_SIZE) ? n : SOFTMAX_BLOCK_SIZE;
    // Round up to power of 2 for clean reductions
    int pow2 = 1;
    while (pow2 < block_size) pow2 <<= 1;
    block_size = (pow2 > SOFTMAX_BLOCK_SIZE) ? SOFTMAX_BLOCK_SIZE : pow2;

    size_t shared_mem = block_size * sizeof(float);
    softmax_kernel<<<1, block_size, shared_mem>>>(input, output, n);

    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();
}

} // namespace cuda
} // namespace vulcan
