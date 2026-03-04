/// @file activation.cu
/// @brief Activation function CUDA kernels — SiLU and ReLU.
///
/// SiLU (Swish) is used in Llama-2's MLP:
///   silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
///
/// ReLU is included for completeness and alternative architectures.


#include "cuda/kernels.h"
#include "cuda/utils.h"
#include <cuda_runtime.h>

namespace vulcan {
namespace cuda {

// ─── CUDA Kernels ───────────────────────────────────────────────────────────

/// SiLU (Sigmoid Linear Unit / Swish) activation.
/// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
///
/// Used in Llama-2 for the gated MLP:
///   hidden = silu(gate_proj(x)) * up_proj(x)
__global__ void silu_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        output[idx] = x / (1.0f + expf(-x));
    }
}

/// ReLU activation: max(0, x)
__global__ void relu_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// ─── Host Wrappers ──────────────────────────────────────────────────────────

void launch_silu(const float* input, float* output, int n) {
    const int block_size = 256;
    const int grid_size = calc_grid_1d(n, block_size);

    silu_kernel<<<grid_size, block_size>>>(input, output, n);

    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();
}

void launch_relu(const float* input, float* output, int n) {
    const int block_size = 256;
    const int grid_size = calc_grid_1d(n, block_size);

    relu_kernel<<<grid_size, block_size>>>(input, output, n);

    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();
}

} // namespace cuda
} // namespace vulcan
