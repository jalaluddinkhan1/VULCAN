/// @file matmul_wmma.cu
/// @brief Tensor Core GEMM — FP16 compute, FP32 accumulate (SM 7.0+).
///
/// Uses the CUDA WMMA (Warp Matrix Multiply-Accumulate) API to drive
/// Tensor Core hardware on NVIDIA Volta / Turing / Ampere / Ada GPUs.
///
/// Why Tensor Cores matter:
///   RTX 2080:  57.7 TFLOPS INT8 / 28.8 TFLOPS FP16 Tensor vs 10.1 TFLOPS FP32 CUDA
///   RTX 3090: 142   TFLOPS INT8 / 71   TFLOPS FP16 Tensor vs 35.6 TFLOPS FP32 CUDA
///   RTX 4090: 330   TFLOPS INT8 / 165  TFLOPS FP16 Tensor vs 82.6 TFLOPS FP32 CUDA
///   ~2-4x speedup over scalar GEMM in practice for transformer-sized matmuls.
///
/// Algorithm:
///   - Each thread block computes a BLOCK_M × BLOCK_N tile of C.
///   - WARPS_M × WARPS_N warps inside the block each own one 16×16 output tile.
///   - K is processed in WMMA_K=16 chunks:
///       1. All 256 threads cooperatively load A and B tiles into shared mem as FP16.
///       2. Each warp loads its fragment via wmma::load_matrix_sync.
///       3. wmma::mma_sync performs Tensor Core multiply-accumulate.
///       4. After all K chunks, the FP32 accumulator is stored to global C.
///
/// Tile layout:
///   WMMA fragment:   16 × 16 × 16  (M × N × K), fixed by CUDA WMMA API for FP16
///   Block tile:      64 × 32  (BLOCK_M × BLOCK_N)
///   Warp grid:        4 ×  2  (WARPS_M × WARPS_N)
///   Threads/block:  256  (8 warps × 32 threads)
///
///   Shared memory:
///     s_A[64][16]  (half)  = 2 KB   — A tile, converted from FP32
///     s_B[16][32]  (half)  = 1 KB   — B tile, converted from FP32
///     c_tmp[8][256] (float)= 8 KB   — boundary store scratch per warp
///     Total: ~11 KB (well within 48 KB per SM)
///
/// Fallback:
///   Matrices with M < 16, N < 16, or K < 16 fall back to the tiled scalar
///   GEMM in matmul.cu to avoid wasted tiles.
///
/// Requires: SM 7.0+ (Volta/Turing/Ampere/Ada). The CMakeLists.txt already
///           targets 70 75 80 86 89 90, so no build changes are needed.

#include "cuda/kernels.h"
#include "cuda/utils.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

namespace vulcan {
namespace cuda {

// ─── Tile Dimensions ────────────────────────────────────────────────────────

/// WMMA fragment size — fixed by the CUDA WMMA API for FP16 inputs.
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

/// Block tile: each block computes BLOCK_M × BLOCK_N elements of C.
constexpr int BLOCK_M = 64;   ///< 4 WMMA tiles along M
constexpr int BLOCK_N = 32;   ///< 2 WMMA tiles along N

/// Warp arrangement within the block.
constexpr int WARPS_M   = BLOCK_M / WMMA_M;   ///< 4 warps in M dimension
constexpr int WARPS_N   = BLOCK_N / WMMA_N;   ///< 2 warps in N dimension
constexpr int NUM_WARPS  = WARPS_M * WARPS_N;  ///< 8 warps per block
constexpr int BLOCK_SIZE = NUM_WARPS * 32;     ///< 256 threads per block

// ─── Kernel ─────────────────────────────────────────────────────────────────

/// Tensor Core GEMM kernel: C = A × B
///
/// @param A   [M × K] row-major, float32
/// @param B   [K × N] row-major, float32
/// @param C   [M × N] row-major, float32 (output)
/// @param M   Rows of A / rows of C
/// @param K   Shared (inner) dimension
/// @param N   Columns of B / columns of C
__global__ void matmul_wmma_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int M, int K, int N) {
    // ── Shared memory ───────────────────────────────────────────────────
    // FP16 tiles for A and B (loaded from FP32 global memory each K-tile).
    // c_tmp provides per-warp scratch for boundary output tiles where we
    // cannot store directly into C without writing out-of-bounds.
    __shared__ half  s_A[BLOCK_M][WMMA_K];           // 64×16 × 2B = 2 KB
    __shared__ half  s_B[WMMA_K][BLOCK_N];           // 16×32 × 2B = 1 KB
    __shared__ float c_tmp[NUM_WARPS][WMMA_M * WMMA_N]; // 8×256 × 4B = 8 KB

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    // Which 16×16 output subtile does this warp own?
    const int warp_row = warp_id / WARPS_N;  // 0 .. WARPS_M-1
    const int warp_col = warp_id % WARPS_N;  // 0 .. WARPS_N-1

    // Top-left corner of this block's tile in global C.
    const int block_row = blockIdx.y * BLOCK_M;
    const int block_col = blockIdx.x * BLOCK_N;

    // Top-left corner of this warp's 16×16 subtile in global C.
    const int c_row = block_row + warp_row * WMMA_M;
    const int c_col = block_col + warp_col * WMMA_N;

    // FP32 accumulator — this lives in registers, not shared memory.
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // ── Main loop: iterate over K dimension in WMMA_K=16 chunks ─────────
    const int num_k_tiles = (K + WMMA_K - 1) / WMMA_K;

    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int k0 = k_tile * WMMA_K;

        // ── Load A tile into shared memory (BLOCK_M × WMMA_K) ───────────
        // All 256 threads participate; each loads (BLOCK_M * WMMA_K / 256) = 4
        // elements. Out-of-bounds positions are padded with 0.
        for (int idx = threadIdx.x; idx < BLOCK_M * WMMA_K; idx += BLOCK_SIZE) {
            const int r = idx / WMMA_K;
            const int c = idx % WMMA_K;
            const int gr = block_row + r;
            const int gk = k0 + c;
            s_A[r][c] = (gr < M && gk < K)
                        ? __float2half(A[gr * K + gk])
                        : __float2half(0.0f);
        }

        // ── Load B tile into shared memory (WMMA_K × BLOCK_N) ───────────
        // Each thread loads (WMMA_K * BLOCK_N / 256) = 2 elements.
        for (int idx = threadIdx.x; idx < WMMA_K * BLOCK_N; idx += BLOCK_SIZE) {
            const int r = idx / BLOCK_N;
            const int c = idx % BLOCK_N;
            const int gk = k0 + r;
            const int gc = block_col + c;
            s_B[r][c] = (gk < K && gc < N)
                        ? __float2half(B[gk * N + gc])
                        : __float2half(0.0f);
        }

        __syncthreads();

        // ── WMMA: load fragments → Tensor Core MMA ───────────────────────
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                       half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                       half, wmma::row_major> b_frag;

        // A fragment: rows [warp_row*16 .. warp_row*16+15], cols [0..15].
        // Leading dimension of s_A is WMMA_K (= 16 columns).
        wmma::load_matrix_sync(a_frag, &s_A[warp_row * WMMA_M][0], WMMA_K);

        // B fragment: rows [0..15], cols [warp_col*16 .. warp_col*16+15].
        // Leading dimension of s_B is BLOCK_N (= 32 columns, row stride).
        wmma::load_matrix_sync(b_frag, &s_B[0][warp_col * WMMA_N], BLOCK_N);

        // Multiply-accumulate on Tensor Cores.
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    // ── Store accumulator to global C ────────────────────────────────────
    if (c_row >= M || c_col >= N) return;  // Entire warp tile is out of bounds

    if (c_row + WMMA_M <= M && c_col + WMMA_N <= N) {
        // ── Fast path: full 16×16 tile is in bounds — store directly. ────
        // wmma::store_matrix_sync writes exactly 16×16 FP32 values.
        wmma::store_matrix_sync(C + c_row * N + c_col, c_frag, N,
                                wmma::mem_row_major);
    } else {
        // ── Boundary path: tile extends outside M or N. ──────────────────
        // Store to per-warp scratch in shared memory, then scatter only the
        // valid elements to global memory via lane-level scatter.
        wmma::store_matrix_sync(c_tmp[warp_id], c_frag, WMMA_N,
                                wmma::mem_row_major);
        __syncwarp();

        for (int idx = lane_id; idx < WMMA_M * WMMA_N; idx += 32) {
            const int r = idx / WMMA_N;
            const int c = idx % WMMA_N;
            if (c_row + r < M && c_col + c < N) {
                C[(c_row + r) * N + (c_col + c)] = c_tmp[warp_id][idx];
            }
        }
    }
}

// ─── Host Wrapper ───────────────────────────────────────────────────────────

/// Launch Tensor Core GEMM.
///
/// Automatically falls back to the scalar tiled GEMM (launch_matmul) when
/// any dimension is smaller than a WMMA tile (< 16), which would waste tiles
/// on pure padding work.
///
/// @param A  [M × K] device pointer, float32
/// @param B  [K × N] device pointer, float32
/// @param C  [M × N] device pointer, float32 (output)
void launch_matmul_wmma(const float* A, const float* B, float* C,
                        int M, int K, int N) {
    // Fall back to scalar GEMM for very small matrices where Tensor Core
    // overhead (FP32→FP16 conversion, fragment load/store) isn't worth it.
    if (M < WMMA_M || N < WMMA_N || K < WMMA_K) {
        launch_matmul(A, B, C, M, K, N);
        return;
    }

    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N,
              (M + BLOCK_M - 1) / BLOCK_M);

    matmul_wmma_kernel<<<grid, block>>>(A, B, C, M, K, N);

    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();
}

} // namespace cuda
} // namespace vulcan
