/// @file matmul.cu
/// @brief Tiled Matrix Multiplication — Custom GEMM kernel.
///
/// Implements C = A * B using shared memory tiling for improved
/// memory access patterns. This is intentionally hand-written
/// (no cuBLAS) to demonstrate understanding of GPU memory hierarchy.


#include "cuda/kernels.h"
#include "cuda/utils.h"
#include <cuda_runtime.h>

namespace vulcan {
namespace cuda {

// ─── Constants ──────────────────────────────────────────────────────────────

/// Tile size for shared memory blocking.
/// Each block loads a TILE_SIZE x TILE_SIZE sub-matrix into shared memory.
constexpr int TILE_SIZE = 32;

// ─── CUDA Kernel ────────────────────────────────────────────────────────────

/// Tiled matrix multiplication kernel.
///
/// Algorithm:
///   1. Each thread block computes a TILE_SIZE x TILE_SIZE tile of C.
///   2. Loop over tiles along the K dimension:
///      a. Load a tile of A and B into shared memory.
///      b. Synchronize threads.
///      c. Compute partial dot products from shared memory.
///      d. Synchronize before loading next tile.
///   3. Write final result to global memory.
///
/// Memory access pattern:
///   - Coalesced loads from global → shared memory
///   - Bank-conflict-free reads from shared memory
///
/// @param A  [M x K] row-major
/// @param B  [K x N] row-major
/// @param C  [M x N] row-major (output)
__global__ void matmul_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int K, int N) {
    // Shared memory tiles
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles along K dimension
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < num_tiles; ++t) {
        // Load tile of A into shared memory
        int a_col = t * TILE_SIZE + threadIdx.x;
        if (row < M && a_col < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile of B into shared memory
        int b_row = t * TILE_SIZE + threadIdx.y;
        if (b_row < K && col < N) {
            tile_B[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product from shared memory
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result to global memory
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ─── Host Wrapper ───────────────────────────────────────────────────────────

void launch_matmul(const float* A, const float* B, float* C,
                   int M, int K, int N) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_kernel<<<grid, block>>>(A, B, C, M, K, N);

    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();
}

} // namespace cuda
} // namespace vulcan
