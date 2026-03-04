#pragma once

/// @file sampler.h
/// @brief VULCAN Sampler — Top-P/Top-K token sampling.
///
/// CPU-side sampling logic for selecting the next token from
/// model output logits. Supports temperature scaling, nucleus
/// (Top-P), and Top-K sampling strategies.

#include <vector>
#include <random>

namespace vulcan {

/// @struct SamplerConfig
/// @brief Parameters controlling sampling behavior.
struct SamplerConfig {
    float temperature = 1.0f;   ///< Temperature for softmax scaling
    float top_p       = 0.9f;   ///< Nucleus sampling threshold
    int   top_k       = 40;     ///< Top-K filtering count
    bool  greedy      = false;  ///< If true, always pick argmax
};

/// @class Sampler
/// @brief Selects the next token from logits using configurable strategies.
///
/// Workflow:
///   1. Apply temperature scaling to logits
///   2. Compute softmax probabilities
///   3. Apply Top-K filtering (keep only K highest)
///   4. Apply Top-P (nucleus) filtering
///   5. Sample from remaining distribution
class Sampler {
public:
    explicit Sampler(const SamplerConfig& config = SamplerConfig{});
    ~Sampler();

    /// Sample the next token from raw logits.
    /// @param logits Raw logit values (size = vocab_size)
    /// @param vocab_size Number of vocabulary entries
    /// @return Selected token ID
    int sample(const float* logits, int vocab_size);

    /// Set the random seed for reproducible sampling.
    void set_seed(uint64_t seed);

    /// Update sampler configuration.
    void set_config(const SamplerConfig& config);

    /// Get current configuration.
    const SamplerConfig& config() const;

private:
    SamplerConfig config_;
    std::mt19937  rng_;

    /// Apply softmax with temperature to logits.
    void softmax(std::vector<float>& probs, float temperature) const;

    /// Apply Top-K filtering: zero out all but top K probabilities.
    void apply_top_k(std::vector<float>& probs, int k) const;

    /// Apply nucleus (Top-P) filtering.
    void apply_top_p(std::vector<float>& probs, float p) const;
};

} // namespace vulcan
