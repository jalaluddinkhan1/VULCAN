/// @file kv_cache.cpp
/// @brief VULCAN KV Cache — GPU memory management for cached K/V tensors.
///
/// Pre-allocates contiguous GPU buffers per layer for maximum sequence length.
/// Supports incremental append (1 token at a time) and batch prefill.
///
/// Memory layout per layer:
///   K: [num_kv_heads, max_seq_len, head_dim] — contiguous row-major
///   V: [num_kv_heads, max_seq_len, head_dim] — contiguous row-major
///
/// On append, the new K/V vectors are written to the correct position
/// using cudaMemcpy (small data, latency-bound, not worth a custom kernel).


#include "vulcan/kv_cache.h"
#include "cuda/utils.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>

namespace vulcan {

KVCache::KVCache(int num_kv_heads, int max_seq_len, int head_dim, int num_layers)
    : num_kv_heads_(num_kv_heads),
      max_seq_len_(max_seq_len),
      head_dim_(head_dim),
      num_layers_(num_layers),
      current_len_(0) {

    size_t per_layer_bytes = static_cast<size_t>(num_kv_heads) * max_seq_len
                           * head_dim * sizeof(float);

    k_cache_.resize(num_layers, nullptr);
    v_cache_.resize(num_layers, nullptr);

    for (int i = 0; i < num_layers; ++i) {
        CUDA_CHECK(cudaMalloc(&k_cache_[i], per_layer_bytes));
        CUDA_CHECK(cudaMalloc(&v_cache_[i], per_layer_bytes));
        // Zero-initialize for safety
        CUDA_CHECK(cudaMemset(k_cache_[i], 0, per_layer_bytes));
        CUDA_CHECK(cudaMemset(v_cache_[i], 0, per_layer_bytes));
    }

    size_t total_mb = (2 * num_layers * per_layer_bytes) / (1024 * 1024);
    std::cout << "[VULCAN] KV Cache allocated: " << num_layers << " layers × "
              << num_kv_heads << " heads × " << max_seq_len << " max_seq × "
              << head_dim << " head_dim = " << total_mb << " MB" << std::endl;
}

KVCache::~KVCache() {
    for (auto ptr : k_cache_) { if (ptr) cudaFree(ptr); }
    for (auto ptr : v_cache_) { if (ptr) cudaFree(ptr); }
}

KVCache::KVCache(KVCache&& other) noexcept
    : num_kv_heads_(other.num_kv_heads_),
      max_seq_len_(other.max_seq_len_),
      head_dim_(other.head_dim_),
      num_layers_(other.num_layers_),
      current_len_(other.current_len_),
      k_cache_(std::move(other.k_cache_)),
      v_cache_(std::move(other.v_cache_)) {
    other.current_len_ = 0;
}

KVCache& KVCache::operator=(KVCache&& other) noexcept {
    if (this != &other) {
        // Free existing
        for (auto ptr : k_cache_) { if (ptr) cudaFree(ptr); }
        for (auto ptr : v_cache_) { if (ptr) cudaFree(ptr); }

        num_kv_heads_ = other.num_kv_heads_;
        max_seq_len_  = other.max_seq_len_;
        head_dim_     = other.head_dim_;
        num_layers_   = other.num_layers_;
        current_len_  = other.current_len_;
        k_cache_      = std::move(other.k_cache_);
        v_cache_      = std::move(other.v_cache_);
        other.current_len_ = 0;
    }
    return *this;
}

void KVCache::append(int layer, const float* k_new, const float* v_new,
                     int start_pos, int num_tokens) {
    if (layer < 0 || layer >= num_layers_) return;
    if (start_pos + num_tokens > max_seq_len_) {
        std::cerr << "[VULCAN] KV Cache overflow: pos=" << start_pos
                  << " + tokens=" << num_tokens
                  << " > max_seq=" << max_seq_len_ << std::endl;
        return;
    }

    // Layout: [num_kv_heads, max_seq_len, head_dim]
    // For each KV head, copy num_tokens rows starting at start_pos
    size_t row_bytes = head_dim_ * sizeof(float);

    for (int h = 0; h < num_kv_heads_; ++h) {
        size_t head_offset = static_cast<size_t>(h) * max_seq_len_ * head_dim_;
        size_t pos_offset  = head_offset + static_cast<size_t>(start_pos) * head_dim_;

        // Source: k_new is [num_tokens, num_kv_heads * head_dim]
        // We need to extract head h's data for each token
        for (int t = 0; t < num_tokens; ++t) {
            size_t src_offset = static_cast<size_t>(t) * num_kv_heads_ * head_dim_
                              + static_cast<size_t>(h) * head_dim_;
            size_t dst_offset = pos_offset + static_cast<size_t>(t) * head_dim_;

            CUDA_CHECK(cudaMemcpy(
                k_cache_[layer] + dst_offset,
                k_new + src_offset,
                row_bytes, cudaMemcpyDeviceToDevice
            ));

            CUDA_CHECK(cudaMemcpy(
                v_cache_[layer] + dst_offset,
                v_new + src_offset,
                row_bytes, cudaMemcpyDeviceToDevice
            ));
        }
    }

    // Update current length if we've extended beyond it
    int new_end = start_pos + num_tokens;
    if (new_end > current_len_) {
        current_len_ = new_end;
    }
}

void KVCache::get(int layer, const float** k_out, const float** v_out) const {
    if (layer >= 0 && layer < num_layers_) {
        *k_out = k_cache_[layer];
        *v_out = v_cache_[layer];
    } else {
        *k_out = nullptr;
        *v_out = nullptr;
    }
}

int KVCache::current_length() const { return current_len_; }
void KVCache::set_length(int len) { current_len_ = len; }
void KVCache::reset() { current_len_ = 0; }
int KVCache::max_seq_len() const { return max_seq_len_; }

size_t KVCache::memory_usage() const {
    return 2ULL * num_layers_ * num_kv_heads_ * max_seq_len_ * head_dim_ * sizeof(float);
}

} // namespace vulcan
