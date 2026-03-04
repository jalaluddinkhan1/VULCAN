/// @file attention.cu
/// @brief Scaled Dot-Product Attention — CUDA kernel.
///
/// Implements: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
///
/// This is the naive O(N^2) implementation suitable for moderate sequence
/// lengths. Can be optimized to FlashAttention v2 with online softmax
/// and tiled computation.

#include "cuda/kernels.h"
#include "cuda/utils.h"
#include <cuda_runtime.h>
#include <cfloat>

namespace vulcan {
namespace cuda {

// ─── CUDA Kernel ────────────────────────────────────────────────────────────

/// Naive scaled dot-product attention kernel.
///
/// Each thread computes one row of the attention output for one
/// batch element and one attention head.
///
/// Algorithm per thread (for query position `row`):
///   1. Compute scores[j] = dot(Q[row], K[j]) for j in [0, seq_len)
///   2. Scale: scores[j] *= 1/sqrt(head_dim)
///   3. Apply causal mask: scores[j] = -inf for j > row
///   4. Softmax over scores
///   5. Output[row] = sum_j(scores[j] * V[j])
///
/// Memory layout (all row-major, contiguous):
///   Q, K, V: [batch, heads, seq_len, head_dim]
///   Output:  [batch, heads, seq_len, head_dim]
///
/// NOTE: This is O(N^2 * d) per head — not suitable for seq_len > 4K.
///       FlashAttention v2 reduces memory from O(N^2) to O(N).
__global__ void attention_kernel(const float* __restrict__ Q,
                                 const float* __restrict__ K,
                                 const float* __restrict__ V,
                                 float* __restrict__ output,
                                 int batch, int heads,
                                 int seq_len, int head_dim) {
    int b   = blockIdx.z;                                    // batch index
    int h   = blockIdx.y;                                    // head index
    int row = blockIdx.x * blockDim.x + threadIdx.x;         // query position

    if (row >= seq_len) return;

    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    // Base offset for this batch/head: [b, h, :, :]
    int bh_offset = ((b * heads + h) * seq_len) * head_dim;

    const float* q_row  = Q + bh_offset + row * head_dim;
    const float* k_base = K + bh_offset;
    const float* v_base = V + bh_offset;
    float*       o_row  = output + bh_offset + row * head_dim;

    // ── Step 1 & 2: Compute scaled attention scores ─────────────────
    // scores[j] = dot(Q[row], K[j]) / sqrt(d_k)
    // We need seq_len scores but can't allocate dynamic arrays,
    // so we compute in two passes: first find max, then softmax + weighted V.

    // Pass 1: Find max score (for numerical stability)
    float max_score = -FLT_MAX;
    for (int j = 0; j <= row; ++j) {  // Causal: only attend to j <= row
        const float* k_row = k_base + j * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += q_row[d] * k_row[d];
        }
        score *= scale;
        if (score > max_score) max_score = score;
    }

    // Pass 2: Compute exp(score - max) and sum
    float sum_exp = 0.0f;
    // We'll also accumulate the weighted V in the same pass below,
    // but first we need the sum for normalization. Third pass for output.
    for (int j = 0; j <= row; ++j) {
        const float* k_row = k_base + j * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += q_row[d] * k_row[d];
        }
        score *= scale;
        sum_exp += expf(score - max_score);
    }

    float inv_sum = 1.0f / sum_exp;

    // ── Step 5: Compute output = sum_j(softmax(score_j) * V[j]) ────
    // Initialize output to zero
    for (int d = 0; d < head_dim; ++d) {
        o_row[d] = 0.0f;
    }

    for (int j = 0; j <= row; ++j) {
        // Recompute score (avoids O(N) memory allocation)
        const float* k_row = k_base + j * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += q_row[d] * k_row[d];
        }
        score *= scale;
        float attn_weight = expf(score - max_score) * inv_sum;

        // Accumulate weighted V
        const float* v_row = v_base + j * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            o_row[d] += attn_weight * v_row[d];
        }
    }
}

// ─── Host Wrapper ───────────────────────────────────────────────────────────

void launch_attention(const float* Q, const float* K, const float* V,
                      float* output,
                      int batch, int heads, int seq_len, int head_dim) {
    const int block_size = 32;
    dim3 block(block_size);
    dim3 grid((seq_len + block_size - 1) / block_size, heads, batch);

    attention_kernel<<<grid, block>>>(Q, K, V, output,
                                     batch, heads, seq_len, head_dim);

    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();
}

} // namespace cuda
} // namespace vulcan
