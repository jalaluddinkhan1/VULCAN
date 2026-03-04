/// @file rope.cu
/// @brief Rotary Position Embedding (RoPE) — CUDA kernel.
///
/// RoPE encodes positional information by rotating pairs of dimensions
/// in the query and key vectors. Used in Llama-2 and most modern LLMs.
///
/// For each pair (x_{2i}, x_{2i+1}) at position pos:
///   x'_{2i}   = x_{2i}   * cos(theta) - x_{2i+1} * sin(theta)
///   x'_{2i+1} = x_{2i}   * sin(theta) + x_{2i+1} * cos(theta)
///
/// where theta = pos / 10000^(2i / d)


#include "cuda/kernels.h"
#include "cuda/utils.h"
#include <cuda_runtime.h>
#include <cmath>

namespace vulcan {
namespace cuda {

// ─── CUDA Kernel ────────────────────────────────────────────────────────────

/// Rotary Position Embedding kernel.
///
/// Applies rotation to pairs of dimensions.
/// Each thread handles one pair (2i, 2i+1).
///
/// @param input   Input tensor [seq_len * head_dim] (device)
/// @param output  Output tensor [seq_len * head_dim] (device)
/// @param pos     Position index for this token
/// @param head_dim Dimension of each head (must be even)
/// @param theta_base  Base for frequency computation (default: 10000.0)
__global__ void rope_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            int pos, int head_dim,
                            float theta_base) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;

    if (idx < half_dim) {
        // Compute rotation frequency for this dimension pair
        float freq = 1.0f / powf(theta_base, 2.0f * idx / static_cast<float>(head_dim));
        float theta = pos * freq;

        float cos_theta = cosf(theta);
        float sin_theta = sinf(theta);

        int i0 = 2 * idx;       // Even dimension
        int i1 = 2 * idx + 1;   // Odd dimension

        float x0 = input[i0];
        float x1 = input[i1];

        // Apply rotation
        output[i0] = x0 * cos_theta - x1 * sin_theta;
        output[i1] = x0 * sin_theta + x1 * cos_theta;
    }
}

// ─── Host Wrapper ───────────────────────────────────────────────────────────

void launch_rope(const float* input, float* output,
                 int pos, int head_dim, float theta_base) {
    int half_dim = head_dim / 2;
    const int block_size = 256;
    const int grid_size = calc_grid_1d(half_dim, block_size);

    rope_kernel<<<grid_size, block_size>>>(input, output, pos, head_dim, theta_base);

    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();
}

} // namespace cuda
} // namespace vulcan
