/// @file engine.cpp
/// @brief VULCAN Engine — Full inference with KV-cached incremental decode.
///
/// Two-phase generation:
///   Prefill: forward(prompt_tokens) → processes all tokens, seeds KV cache
///   Decode:  forward_one(token, pos) → processes 1 token, extends KV cache
///
/// This eliminates O(N²) recomputation — each decode step is O(N) for
/// attention over the cached sequence, O(1) for all other operations.


#include "vulcan/engine.h"
#include "cuda/kernels.h"
#include "cuda/utils.h"
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

namespace vulcan {

Engine::Engine()
    : sampler_(), ready_(false) {}

Engine::~Engine() = default;

bool Engine::load_model(const std::string& path, const ModelConfig& config) {
    std::cout << "[VULCAN] Loading model from: " << path << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    bool success = model_.load(path, config);
    auto end = std::chrono::high_resolution_clock::now();

    if (!success) {
        std::cerr << "[VULCAN] Failed to load model." << std::endl;
        ready_ = false;
        return false;
    }

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "[VULCAN] Model loaded in " << ms.count() << " ms" << std::endl;
    model_.print_summary();

    // Initialize transformer layers
    layers_.clear();
    layers_.reserve(config.num_layers);
    for (int i = 0; i < config.num_layers; ++i) {
        layers_.emplace_back(
            std::make_unique<TransformerBlock>(model_, config, i)
        );
    }

    // Allocate KV cache
    int head_dim = config.hidden_dim / config.num_heads;
    kv_cache_ = std::make_unique<KVCache>(
        config.num_kv_heads,
        2048,            // max_seq_len (configurable via GenerationConfig)
        head_dim,
        config.num_layers
    );

    std::cout << "[VULCAN] KV Cache: " << (cache_memory_usage() / (1024*1024))
              << " MB allocated" << std::endl;

    ready_ = true;
    return true;
}

// ─── Prefill Forward Pass ───────────────────────────────────────────────────

std::vector<float> Engine::forward(const std::vector<int>& input_ids) {
    if (!ready_) {
        std::cerr << "[VULCAN] Error: Model not loaded." << std::endl;
        return {};
    }

    const auto& config = model_.config();
    const int seq_len    = static_cast<int>(input_ids.size());
    const int hidden_dim = config.hidden_dim;

    // Reset KV cache for fresh prefill
    kv_cache_->reset();

    // ── 1. Embedding lookup ─────────────────────────────────────────
    int* d_token_ids;
    CUDA_CHECK(cudaMalloc(&d_token_ids, seq_len * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_token_ids, input_ids.data(),
                          seq_len * sizeof(int), cudaMemcpyHostToDevice));

    const Tensor& embed_table = model_.get_weight("tok_embeddings.weight");
    Tensor hidden({seq_len, hidden_dim}, Device::CUDA);
    cuda::launch_embedding_lookup(embed_table.data(), d_token_ids,
                                  hidden.data(), seq_len, hidden_dim);
    cudaFree(d_token_ids);

    // ── 2. Through all layers (prefill: all tokens at once) ─────────
    hidden = run_layers(hidden, seq_len, 0);
    kv_cache_->set_length(seq_len);

    // ── 3. Logit projection ─────────────────────────────────────────
    return logit_projection(hidden, seq_len);
}

// ─── Incremental Decode (1 token) ───────────────────────────────────────────

std::vector<float> Engine::forward_one(int token_id, int pos) {
    if (!ready_) return {};

    const auto& config = model_.config();
    const int hidden_dim = config.hidden_dim;

    // ── 1. Embed single token ───────────────────────────────────────
    int* d_token_id;
    CUDA_CHECK(cudaMalloc(&d_token_id, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_token_id, &token_id, sizeof(int), cudaMemcpyHostToDevice));

    const Tensor& embed_table = model_.get_weight("tok_embeddings.weight");
    Tensor hidden({1, hidden_dim}, Device::CUDA);
    cuda::launch_embedding_lookup(embed_table.data(), d_token_id,
                                  hidden.data(), 1, hidden_dim);
    cudaFree(d_token_id);

    // ── 2. Through all layers (decode: 1 token, extend KV cache) ───
    hidden = run_layers(hidden, 1, pos);

    // ── 3. Logit projection ─────────────────────────────────────────
    return logit_projection(hidden, 1);
}

// ─── Generation Loop ────────────────────────────────────────────────────────

std::vector<int> Engine::generate(const std::vector<int>& prompt_ids,
                                  const GenerationConfig& gen_config) {
    if (!ready_) {
        std::cerr << "[VULCAN] Error: Model not loaded." << std::endl;
        return prompt_ids;
    }

    // Configure sampler
    SamplerConfig samp_cfg;
    samp_cfg.temperature = gen_config.temperature;
    samp_cfg.top_p       = gen_config.top_p;
    samp_cfg.top_k       = gen_config.top_k;
    samp_cfg.greedy      = gen_config.greedy;
    sampler_.set_config(samp_cfg);

    const int vocab_size  = model_.config().vocab_size;
    const int prompt_len  = static_cast<int>(prompt_ids.size());
    std::vector<int> tokens = prompt_ids;

    std::cout << "[VULCAN] Prefill: " << prompt_len << " tokens..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    // ── Prefill ──────────────────────────────────────────────────────
    // Process entire prompt, populate KV cache
    std::vector<float> logits = forward(prompt_ids);
    if (logits.empty()) return prompt_ids;

    auto prefill_end = std::chrono::high_resolution_clock::now();
    auto prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        prefill_end - start);
    float prefill_tps = (prompt_len > 0 && prefill_ms.count() > 0)
        ? (prompt_len * 1000.0f / prefill_ms.count()) : 0.0f;
    std::cout << "[VULCAN] Prefill: " << prefill_ms.count() << " ms ("
              << prefill_tps << " tok/s)" << std::endl;

    // ── Decode ───────────────────────────────────────────────────────
    // Generate tokens one at a time using KV cache
    std::cout << "[VULCAN] Decode: generating up to " << gen_config.max_tokens
              << " tokens..." << std::endl;

    auto decode_start = std::chrono::high_resolution_clock::now();
    int generated = 0;

    for (int t = 0; t < gen_config.max_tokens; ++t) {
        // Sample next token from logits
        int next_token = sampler_.sample(logits.data(), vocab_size);

        if (next_token == gen_config.eos_token_id) {
            std::cout << "[VULCAN] EOS at step " << t << std::endl;
            break;
        }

        tokens.push_back(next_token);
        generated++;

        // Check sequence length limit
        int current_pos = prompt_len + generated;
        if (current_pos >= gen_config.max_seq_len) {
            std::cout << "[VULCAN] Max sequence length reached." << std::endl;
            break;
        }

        // Incremental decode: forward only the new token
        logits = forward_one(next_token, current_pos - 1);
        if (logits.empty()) break;
    }

    auto decode_end = std::chrono::high_resolution_clock::now();
    auto decode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        decode_end - decode_start);
    float decode_tps = (generated > 0 && decode_ms.count() > 0)
        ? (generated * 1000.0f / decode_ms.count()) : 0.0f;

    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        decode_end - start);

    std::cout << "[VULCAN] Decode: " << generated << " tokens in "
              << decode_ms.count() << " ms (" << decode_tps << " tok/s)\n"
              << "[VULCAN] Total: " << total_ms.count() << " ms\n"
              << "[VULCAN] KV Cache: " << (cache_memory_usage() / (1024*1024))
              << " MB used" << std::endl;

    return tokens;
}

// ─── Internal Helpers ───────────────────────────────────────────────────────

Tensor Engine::run_layers(const Tensor& hidden, int num_tokens, int start_pos) {
    Tensor h = hidden.to(Device::CUDA);
    for (int i = 0; i < static_cast<int>(layers_.size()); ++i) {
        h = layers_[i]->forward(h, num_tokens, start_pos, kv_cache_.get());
    }
    return h;
}

std::vector<float> Engine::logit_projection(const Tensor& hidden, int num_tokens) {
    const auto& config = model_.config();
    const int hidden_dim = config.hidden_dim;
    const int vocab_size = config.vocab_size;

    // Final RMSNorm
    const Tensor& final_norm_w = model_.get_weight("norm.weight");
    Tensor normed({num_tokens, hidden_dim}, Device::CUDA);

    for (int t = 0; t < num_tokens; ++t) {
        cuda::launch_rmsnorm(
            hidden.data() + t * hidden_dim,
            final_norm_w.data(),
            normed.data() + t * hidden_dim,
            hidden_dim,
            config.norm_eps
        );
    }

    // Logit = hidden[-1] @ output_weight^T → [vocab_size]
    const Tensor& output_weight = model_.get_weight("output.weight");
    Tensor logits_tensor({1, vocab_size}, Device::CUDA);

    cuda::launch_matmul(
        normed.data() + (num_tokens - 1) * hidden_dim,
        output_weight.data(),
        logits_tensor.data(),
        1, hidden_dim, vocab_size
    );

    std::vector<float> logits(vocab_size);
    logits_tensor.to_host(logits.data());
    return logits;
}

void Engine::reset_cache() {
    if (kv_cache_) kv_cache_->reset();
}

bool Engine::is_ready() const { return ready_; }

size_t Engine::cache_memory_usage() const {
    return kv_cache_ ? kv_cache_->memory_usage() : 0;
}

} // namespace vulcan
