#pragma once

/// @file transformer.h
/// @brief VULCAN TransformerBlock — Single Llama-2 transformer layer.
///
/// Implements the full Llama-2 transformer layer with KV cache support:
///   1. Pre-attention RMSNorm
///   2. Multi-head self-attention (Q/K/V projections + RoPE + cached attention)
///   3. Residual connection
///   4. Pre-FFN RMSNorm
///   5. Gated MLP (gate_proj, up_proj → SiLU gate * up → down_proj)
///   6. Residual connection
///


#include "vulcan/tensor.h"
#include "vulcan/model.h"
#include "vulcan/kv_cache.h"
#include <string>

namespace vulcan {

/// @class TransformerBlock
/// @brief Executes one transformer layer of the Llama-2 model.
///
/// Supports KV cache for incremental decode.
/// During prefill, all tokens are processed and K/V stored in cache.
/// During decode, only the latest token is processed, reusing cached K/V.
class TransformerBlock {
public:
    TransformerBlock(const Model& model, const ModelConfig& config, int layer);
    ~TransformerBlock();

    /// Run the transformer block with KV cache support.
    ///
    /// @param hidden      Input hidden states [num_tokens, hidden_dim] on GPU
    /// @param num_tokens  Number of tokens being processed (seq_len for prefill, 1 for decode)
    /// @param start_pos   Position of first token in the sequence
    /// @param kv_cache    KV cache to read/write cached keys and values
    /// @return Output hidden states [num_tokens, hidden_dim] on GPU
    Tensor forward(const Tensor& hidden, int num_tokens, int start_pos,
                   KVCache* kv_cache = nullptr);

private:
    const Model&       model_;
    const ModelConfig& config_;
    int                layer_;

    std::string weight_name(const std::string& suffix) const;

    /// Run self-attention with KV cache.
    Tensor self_attention(const Tensor& normed_input, int num_tokens,
                          int start_pos, KVCache* kv_cache);

    /// Run gated MLP sub-block.
    Tensor mlp(const Tensor& normed_input, int num_tokens);
};

} // namespace vulcan
