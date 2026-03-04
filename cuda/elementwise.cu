/// @file elementwise.cu
/// @brief Element-wise CUDA kernels — add, multiply, residual.
///
/// Basic building blocks for transformer forward pass:
///   - Vector addition (residual connections)
///   - Element-wise multiply (gated MLP)
///   - Embedding lookup

#include "cuda/kernels.h"
#include "cuda/utils.h"
#include <cuda_runtime.h>

namespace vulcan {
namespace cuda {

// ─── Residual Add ───────────────────────────────────────────────────────────

/// c[i] = a[i] + b[i]  (residual connection)
__global__ void residual_add_kernel(const float* __restrict__ a,
                                    const float* __restrict__ b,
                                    float* __restrict__ c,
                                    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void launch_residual_add(const float* a, const float* b, float* c, int n) {
    const int block_size = 256;
    const int grid_size = calc_grid_1d(n, block_size);
    residual_add_kernel<<<grid_size, block_size>>>(a, b, c, n);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();
}

// ─── Element-wise Multiply ──────────────────────────────────────────────────

/// c[i] = a[i] * b[i]  (gated MLP: silu(gate) * up)
__global__ void elementwise_mul_kernel(const float* __restrict__ a,
                                       const float* __restrict__ b,
                                       float* __restrict__ c,
                                       int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

void launch_elementwise_mul(const float* a, const float* b, float* c, int n) {
    const int block_size = 256;
    const int grid_size = calc_grid_1d(n, block_size);
    elementwise_mul_kernel<<<grid_size, block_size>>>(a, b, c, n);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();
}

// ─── Embedding Lookup ───────────────────────────────────────────────────────

/// For each token ID, copy the corresponding row from the embedding table.
/// output[i * dim ... (i+1)*dim) = table[token_ids[i] * dim ... ]
__global__ void embedding_lookup_kernel(const float* __restrict__ table,
                                        const int* __restrict__ token_ids,
                                        float* __restrict__ output,
                                        int seq_len, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int token_idx = idx / dim;
    int dim_idx = idx % dim;

    if (token_idx < seq_len) {
        int token_id = token_ids[token_idx];
        output[token_idx * dim + dim_idx] = table[token_id * dim + dim_idx];
    }
}

void launch_embedding_lookup(const float* table, const int* token_ids,
                             float* output, int seq_len, int dim) {
    int total = seq_len * dim;
    const int block_size = 256;
    const int grid_size = calc_grid_1d(total, block_size);
    embedding_lookup_kernel<<<grid_size, block_size>>>(
        table, token_ids, output, seq_len, dim);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();
}

} // namespace cuda
} // namespace vulcan
