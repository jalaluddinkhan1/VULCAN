#pragma once

/// @file kv_cache.h
/// @brief VULCAN KV Cache — Key-Value cache for incremental decode.
///
/// During autoregressive generation, the attention mechanism reuses
/// previously computed K and V matrices. Without caching, each new
/// token requires recomputing K/V for the entire sequence (O(N²)).
/// With KV cache, only the new token's K/V is computed (O(N)).
///
/// Design:
///   - Pre-allocated contiguous GPU buffers for K and V per layer per head
///   - Circular write pointer (appends new positions)
///   - Supports incremental decode (1 new token per step)
///   - Supports prefill (batch-write entire prompt in one shot)


#include <vector>
#include <cstddef>

namespace vulcan {

/// @class KVCache
/// @brief Per-layer Key-Value cache for multi-head attention.
///
/// Stores K and V tensors in pre-allocated GPU memory:
///   K cache: [num_kv_heads, max_seq_len, head_dim]
///   V cache: [num_kv_heads, max_seq_len, head_dim]
///
/// Usage:
///   KVCache cache(32, 2048, 128, 8);  // 32 heads, max 2048 tokens, 128 head_dim
///   cache.append(new_k, new_v, current_pos, seq_len=1);
///   auto [k, v, len] = cache.get();  // returns pointers + current length
class KVCache {
public:
    /// @param num_kv_heads  Number of KV heads (may differ from Q heads in GQA)
    /// @param max_seq_len   Maximum sequence length (pre-allocated)
    /// @param head_dim      Dimension per head
    /// @param num_layers    Number of transformer layers
    KVCache(int num_kv_heads, int max_seq_len, int head_dim, int num_layers);
    ~KVCache();

    // Move-only
    KVCache(KVCache&& other) noexcept;
    KVCache& operator=(KVCache&& other) noexcept;
    KVCache(const KVCache&) = delete;
    KVCache& operator=(const KVCache&) = delete;

    /// Append new K and V vectors to the cache for a given layer.
    ///
    /// @param layer     Layer index
    /// @param k_new     New K vectors [num_tokens, num_kv_heads * head_dim] on GPU
    /// @param v_new     New V vectors [num_tokens, num_kv_heads * head_dim] on GPU
    /// @param start_pos Position to start writing (0 for first token)
    /// @param num_tokens Number of tokens to write (1 for incremental, N for prefill)
    void append(int layer, const float* k_new, const float* v_new,
                int start_pos, int num_tokens);

    /// Get pointers to the cached K and V for a given layer.
    ///
    /// @param layer    Layer index
    /// @param k_out    Output: pointer to K cache [num_kv_heads, current_len, head_dim]
    /// @param v_out    Output: pointer to V cache [num_kv_heads, current_len, head_dim]
    void get(int layer, const float** k_out, const float** v_out) const;

    /// Get the current sequence length in the cache.
    int current_length() const;

    /// Set the current sequence length (after prefill/append).
    void set_length(int len);

    /// Reset the cache (for new generation).
    void reset();

    /// Get the maximum sequence length.
    int max_seq_len() const;

    /// Get total GPU memory used by this cache (bytes).
    size_t memory_usage() const;

private:
    int num_kv_heads_;
    int max_seq_len_;
    int head_dim_;
    int num_layers_;
    int current_len_;  ///< How many positions are currently filled

    /// GPU memory for K and V caches, indexed by layer.
    /// Each entry is a contiguous buffer of [num_kv_heads * max_seq_len * head_dim] floats.
    std::vector<float*> k_cache_;  ///< k_cache_[layer] → GPU pointer
    std::vector<float*> v_cache_;  ///< v_cache_[layer] → GPU pointer
};

} // namespace vulcan
