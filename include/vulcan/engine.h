#pragma once

/// @file engine.h
/// @brief VULCAN Engine — Main inference orchestrator.
///
/// Supports KV cache for O(1) per-token decode:
///   - Prefill: process entire prompt, populate KV cache
///   - Decode:  process 1 token per step, reuse cached K/V

#include "vulcan/model.h"
#include "vulcan/sampler.h"
#include "vulcan/transformer.h"
#include "vulcan/kv_cache.h"
#include <string>
#include <vector>
#include <memory>

namespace vulcan {

/// @struct GenerationConfig
/// @brief Configuration for text generation.
struct GenerationConfig {
    int   max_tokens    = 256;    ///< Maximum tokens to generate
    int   max_seq_len   = 2048;   ///< Maximum total sequence length (prompt + generated)
    float temperature   = 1.0f;   ///< Sampling temperature
    float top_p         = 0.9f;   ///< Nucleus sampling threshold
    int   top_k         = 40;     ///< Top-K sampling count
    bool  greedy        = false;  ///< Greedy decoding
    int   eos_token_id  = 2;      ///< End-of-sequence token ID
};

/// @class Engine
/// @brief High-level inference engine with KV cache for efficient generation.
///
/// Generation loop:
///   1. Prefill:  forward(prompt)     → process all prompt tokens, fill KV cache
///   2. Decode:   forward_one(token)  → process single token, extend KV cache
///   3. Repeat decode until EOS or max_tokens
class Engine {
public:
    Engine();
    ~Engine();

    /// Load model and allocate KV cache.
    bool load_model(const std::string& path, const ModelConfig& config);

    /// Full forward pass (prefill mode): processes entire sequence.
    /// @param input_ids  Full token sequence
    /// @return Logits for the last position [vocab_size]
    std::vector<float> forward(const std::vector<int>& input_ids);

    /// Single-token forward pass (decode mode): processes one new token.
    /// Requires KV cache to be populated via a prior forward() call.
    /// @param token_id   Single token to process
    /// @param pos        Position of this token in the full sequence
    /// @return Logits [vocab_size]
    std::vector<float> forward_one(int token_id, int pos);

    /// Generate tokens with KV-cached incremental decode.
    /// @param prompt_ids  Input prompt token IDs
    /// @param gen_config  Generation parameters
    /// @return Generated token IDs (including prompt)
    std::vector<int> generate(const std::vector<int>& prompt_ids,
                              const GenerationConfig& gen_config = GenerationConfig{});

    /// Reset KV cache (for new generation).
    void reset_cache();

    bool is_ready() const;

    /// Get KV cache memory usage in bytes.
    size_t cache_memory_usage() const;

private:
    Model                                        model_;
    Sampler                                      sampler_;
    bool                                         ready_;
    std::vector<std::unique_ptr<TransformerBlock>> layers_;
    std::unique_ptr<KVCache>                     kv_cache_;

    /// Internal: run hidden states through all layers.
    Tensor run_layers(const Tensor& hidden, int num_tokens, int start_pos);

    /// Internal: final norm + logit projection for last token.
    std::vector<float> logit_projection(const Tensor& hidden, int num_tokens);
};

} // namespace vulcan
