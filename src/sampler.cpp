/// @file sampler.cpp
/// @brief VULCAN Sampler — Token sampling implementation.

#include "vulcan/sampler.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cassert>

namespace vulcan {

Sampler::Sampler(const SamplerConfig& config)
    : config_(config), rng_(42) {}  // Default seed: 42

Sampler::~Sampler() = default;

int Sampler::sample(const float* logits, int vocab_size) {
    assert(logits && "Logits pointer must not be null");
    assert(vocab_size > 0 && "Vocabulary size must be positive");

    // Greedy: just return argmax
    if (config_.greedy) {
        return static_cast<int>(
            std::max_element(logits, logits + vocab_size) - logits
        );
    }

    // Copy logits to probability vector
    std::vector<float> probs(logits, logits + vocab_size);

    // Apply temperature + softmax
    softmax(probs, config_.temperature);

    // Apply Top-K filtering
    if (config_.top_k > 0 && config_.top_k < vocab_size) {
        apply_top_k(probs, config_.top_k);
    }

    // Apply Top-P (nucleus) filtering
    if (config_.top_p > 0.0f && config_.top_p < 1.0f) {
        apply_top_p(probs, config_.top_p);
    }

    // Renormalize after filtering
    float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
    if (sum > 0.0f) {
        for (float& p : probs) p /= sum;
    }

    // Sample from the filtered distribution
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return dist(rng_);
}

void Sampler::set_seed(uint64_t seed) {
    rng_.seed(static_cast<std::mt19937::result_type>(seed));
}

void Sampler::set_config(const SamplerConfig& config) {
    config_ = config;
}

const SamplerConfig& Sampler::config() const {
    return config_;
}

// ─── Private Helpers ────────────────────────────────────────────────────────

void Sampler::softmax(std::vector<float>& probs, float temperature) const {
    assert(temperature > 0.0f && "Temperature must be positive");

    // Scale by temperature
    float inv_temp = 1.0f / temperature;
    for (float& p : probs) p *= inv_temp;

    // Subtract max for numerical stability
    float max_val = *std::max_element(probs.begin(), probs.end());
    float sum = 0.0f;
    for (float& p : probs) {
        p = std::exp(p - max_val);
        sum += p;
    }

    // Normalize
    for (float& p : probs) p /= sum;
}

void Sampler::apply_top_k(std::vector<float>& probs, int k) const {
    // Find the k-th largest probability
    std::vector<float> sorted_probs = probs;
    std::partial_sort(sorted_probs.begin(),
                      sorted_probs.begin() + k,
                      sorted_probs.end(),
                      std::greater<float>());

    float threshold = sorted_probs[k - 1];

    // Zero out everything below the k-th value
    for (float& p : probs) {
        if (p < threshold) p = 0.0f;
    }
}

void Sampler::apply_top_p(std::vector<float>& probs, float p) const {
    // Sort indices by probability (descending)
    std::vector<int> indices(probs.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&probs](int a, int b) { return probs[a] > probs[b]; });

    // Accumulate probabilities until we exceed p
    float cumsum = 0.0f;
    int cutoff_idx = static_cast<int>(indices.size());

    for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
        cumsum += probs[indices[i]];
        if (cumsum > p) {
            cutoff_idx = i + 1;  // Include this token
            break;
        }
    }

    // Zero out everything beyond the cutoff
    for (int i = cutoff_idx; i < static_cast<int>(indices.size()); ++i) {
        probs[indices[i]] = 0.0f;
    }
}

} // namespace vulcan
