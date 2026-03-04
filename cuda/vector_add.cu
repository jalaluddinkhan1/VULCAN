/// @file vector_add.cu
/// @brief Verification kernel — element-wise vector addition.
///
/// This is the first CUDA kernel in VULCAN. Its purpose is to verify
/// the build pipeline: CMake finds CUDA, compiles .cu files, and
/// the kernel launches correctly on the GPU.

#include "cuda/kernels.h"
#include "cuda/utils.h"
#include <cuda_runtime.h>

namespace vulcan {
namespace cuda {

// ─── CUDA Kernel ────────────────────────────────────────────────────────────

/// Element-wise addition kernel.
/// Each thread computes one output element: c[i] = a[i] + b[i]
///
/// Grid:  1D, ceil(n / blockDim.x) blocks
/// Block: 1D, 256 threads (default)
__global__ void vector_add_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c,
                                  int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// ─── Host Wrapper ───────────────────────────────────────────────────────────

void launch_vector_add(const float* a, const float* b, float* c, int n) {
    const int block_size = 256;
    const int grid_size = calc_grid_1d(n, block_size);

    vector_add_kernel<<<grid_size, block_size>>>(a, b, c, n);

    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();
}

} // namespace cuda
} // namespace vulcan
