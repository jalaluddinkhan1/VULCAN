/// @file transformer.cpp
/// @brief VULCAN TransformerBlock — Llama-2 layer with KV cache support.
///
/// self_attention reads/writes the KV cache:
///   - Prefill: computes Q/K/V for all tokens, stores K/V in cache
///   - Decode:  computes Q/K/V for 1 token, appends to cache, attends to full cache

#include "vulcan/transformer.h"
#include "cuda/kernels.h"
#include "cuda/utils.h"
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>

namespace vulcan {

TransformerBlock::TransformerBlock(const Model& model,
                                  const ModelConfig& config,
                                  int layer)
    : model_(model), config_(config), layer_(layer) {}

TransformerBlock::~TransformerBlock() = default;

std::string TransformerBlock::weight_name(const std::string& suffix) const {
    std::ostringstream oss;
    oss << "layers." << layer_ << "." << suffix;
    return oss.str();
}

// ─── Forward Pass ───────────────────────────────────────────────────────────

Tensor TransformerBlock::forward(const Tensor& hidden, int num_tokens,
                                 int start_pos, KVCache* kv_cache) {
    const int hidden_dim = config_.hidden_dim;
    const int total_elems = num_tokens * hidden_dim;

    // ── 1. Pre-attention RMSNorm ────────────────────────────────────
    Tensor normed_attn(hidden.shape(), Device::CUDA);
    const Tensor& attn_norm_w = model_.get_weight(weight_name("attention_norm.weight"));

    for (int t = 0; t < num_tokens; ++t) {
        cuda::launch_rmsnorm(
            hidden.data() + t * hidden_dim,
            attn_norm_w.data(),
            normed_attn.data() + t * hidden_dim,
            hidden_dim,
            config_.norm_eps
        );
    }

    // ── 2. Self-Attention (with KV cache) ───────────────────────────
    Tensor attn_out = self_attention(normed_attn, num_tokens, start_pos, kv_cache);

    // ── 3. Residual ─────────────────────────────────────────────────
    Tensor post_attn({num_tokens, hidden_dim}, Device::CUDA);
    cuda::launch_residual_add(hidden.data(), attn_out.data(),
                              post_attn.data(), total_elems);

    // ── 4. Pre-FFN RMSNorm ──────────────────────────────────────────
    Tensor normed_ffn(post_attn.shape(), Device::CUDA);
    const Tensor& ffn_norm_w = model_.get_weight(weight_name("ffn_norm.weight"));

    for (int t = 0; t < num_tokens; ++t) {
        cuda::launch_rmsnorm(
            post_attn.data() + t * hidden_dim,
            ffn_norm_w.data(),
            normed_ffn.data() + t * hidden_dim,
            hidden_dim,
            config_.norm_eps
        );
    }

    // ── 5. MLP ──────────────────────────────────────────────────────
    Tensor mlp_out = mlp(normed_ffn, num_tokens);

    // ── 6. Residual ─────────────────────────────────────────────────
    Tensor output({num_tokens, hidden_dim}, Device::CUDA);
    cuda::launch_residual_add(post_attn.data(), mlp_out.data(),
                              output.data(), total_elems);

    return output;
}

// ─── Self-Attention with KV Cache ───────────────────────────────────────────

Tensor TransformerBlock::self_attention(const Tensor& normed_input,
                                       int num_tokens, int start_pos,
                                       KVCache* kv_cache) {
    const int hidden_dim    = config_.hidden_dim;
    const int num_heads     = config_.num_heads;
    const int num_kv_heads  = config_.num_kv_heads;
    const int head_dim      = hidden_dim / num_heads;
    const int kv_dim        = num_kv_heads * head_dim;

    // Get weight tensors
    const Tensor& wq = model_.get_weight(weight_name("attention.wq.weight"));
    const Tensor& wk = model_.get_weight(weight_name("attention.wk.weight"));
    const Tensor& wv = model_.get_weight(weight_name("attention.wv.weight"));
    const Tensor& wo = model_.get_weight(weight_name("attention.wo.weight"));

    // ── Q/K/V Projections ───────────────────────────────────────────
    Tensor Q({num_tokens, hidden_dim}, Device::CUDA);
    cuda::launch_matmul(normed_input.data(), wq.data(), Q.data(),
                        num_tokens, hidden_dim, hidden_dim);

    Tensor K_new({num_tokens, kv_dim}, Device::CUDA);
    cuda::launch_matmul(normed_input.data(), wk.data(), K_new.data(),
                        num_tokens, hidden_dim, kv_dim);

    Tensor V_new({num_tokens, kv_dim}, Device::CUDA);
    cuda::launch_matmul(normed_input.data(), wv.data(), V_new.data(),
                        num_tokens, hidden_dim, kv_dim);

    // ── Apply RoPE ──────────────────────────────────────────────────
    for (int t = 0; t < num_tokens; ++t) {
        int pos = start_pos + t;
        for (int h = 0; h < num_heads; ++h) {
            int q_offset = t * hidden_dim + h * head_dim;
            cuda::launch_rope(Q.data() + q_offset,
                              Q.data() + q_offset,
                              pos, head_dim, config_.rope_theta);
        }
        for (int h = 0; h < num_kv_heads; ++h) {
            int k_offset = t * kv_dim + h * head_dim;
            cuda::launch_rope(K_new.data() + k_offset,
                              K_new.data() + k_offset,
                              pos, head_dim, config_.rope_theta);
        }
    }

    // ── KV Cache: append new K/V and get full cached sequence ───────
    const float* k_for_attn;
    const float* v_for_attn;
    int attn_seq_len;

    if (kv_cache) {
        // Store new K/V in cache
        kv_cache->append(layer_, K_new.data(), V_new.data(),
                         start_pos, num_tokens);

        // Get full cached K/V
        kv_cache->get(layer_, &k_for_attn, &v_for_attn);
        attn_seq_len = kv_cache->current_length();
    } else {
        // No cache — use K/V from current tokens only (prefill fallback)
        k_for_attn = K_new.data();
        v_for_attn = V_new.data();
        attn_seq_len = num_tokens;
    }

    // ── Multi-Head Attention ────────────────────────────────────────
    // With KV cache, Q has shape [num_tokens, ...] but K/V have [attn_seq_len, ...]
    // The attention kernel handles causal masking internally.

    Tensor attn_output({num_tokens, hidden_dim}, Device::CUDA);

    if (num_kv_heads == num_heads) {
        // Standard MHA — but need to handle Q_seq != K/V_seq
        // Launch per-head attention with cached K/V
        for (int h = 0; h < num_heads; ++h) {
            // Q: offset into [num_tokens, num_heads, head_dim] layout
            // For head h, Q is at positions [t * hidden_dim + h * head_dim] for each token t
            // But attention kernel expects contiguous [1, 1, seq_len, head_dim]
            // Since we're doing incremental decode (typically num_tokens=1), this is efficient
            cuda::launch_attention(
                Q.data() + h * head_dim,   // Q for head h
                k_for_attn + h * kv_cache->max_seq_len() * head_dim,  // cached K
                v_for_attn + h * kv_cache->max_seq_len() * head_dim,  // cached V
                attn_output.data() + h * head_dim,
                1, 1, attn_seq_len, head_dim
            );
        }
    } else {
        // GQA
        int heads_per_kv = num_heads / num_kv_heads;
        int max_seq = kv_cache ? kv_cache->max_seq_len() : attn_seq_len;
        for (int kv_h = 0; kv_h < num_kv_heads; ++kv_h) {
            for (int rep = 0; rep < heads_per_kv; ++rep) {
                int q_head = kv_h * heads_per_kv + rep;
                cuda::launch_attention(
                    Q.data() + q_head * head_dim,
                    k_for_attn + kv_h * max_seq * head_dim,
                    v_for_attn + kv_h * max_seq * head_dim,
                    attn_output.data() + q_head * head_dim,
                    1, 1, attn_seq_len, head_dim
                );
            }
        }
    }

    // ── Output Projection ───────────────────────────────────────────
    Tensor output({num_tokens, hidden_dim}, Device::CUDA);
    cuda::launch_matmul(attn_output.data(), wo.data(), output.data(),
                        num_tokens, hidden_dim, hidden_dim);

    return output;
}

// ─── Gated MLP ──────────────────────────────────────────────────────────────

Tensor TransformerBlock::mlp(const Tensor& normed_input, int num_tokens) {
    const int hidden_dim    = config_.hidden_dim;
    const int intermediate  = config_.intermediate;

    const Tensor& w1 = model_.get_weight(weight_name("feed_forward.w1.weight"));
    const Tensor& w2 = model_.get_weight(weight_name("feed_forward.w2.weight"));
    const Tensor& w3 = model_.get_weight(weight_name("feed_forward.w3.weight"));

    int inter_total = num_tokens * intermediate;

    Tensor gate({num_tokens, intermediate}, Device::CUDA);
    cuda::launch_matmul(normed_input.data(), w1.data(), gate.data(),
                        num_tokens, hidden_dim, intermediate);

    Tensor up({num_tokens, intermediate}, Device::CUDA);
    cuda::launch_matmul(normed_input.data(), w3.data(), up.data(),
                        num_tokens, hidden_dim, intermediate);

    // Fused SiLU(gate) * up — eliminates intermediate buffer + 1 kernel launch
    // (See ADR-002 and tests/test_quantization.cu FusedSiLUMulTest)
    Tensor gated({num_tokens, intermediate}, Device::CUDA);
    cuda::launch_fused_silu_mul(gate.data(), up.data(),
                                gated.data(), inter_total);

    Tensor output({num_tokens, hidden_dim}, Device::CUDA);
    cuda::launch_matmul(gated.data(), w2.data(), output.data(),
                        num_tokens, intermediate, hidden_dim);

    return output;
}

} // namespace vulcan
