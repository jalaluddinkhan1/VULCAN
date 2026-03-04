#pragma once

/// @file model.h
/// @brief VULCAN Model — Weight loading and model structure.
///
/// Handles loading model weights from binary files into GPU tensors.
/// Supports Llama-2 architecture with plans for GGUF/SafeTensors parsing.

#include "vulcan/tensor.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace vulcan {

/// @struct ModelConfig
/// @brief Configuration parameters for the transformer model.
struct ModelConfig {
    int vocab_size    = 32000;   ///< Vocabulary size
    int hidden_dim    = 4096;    ///< Hidden dimension (d_model)
    int num_heads     = 32;      ///< Number of attention heads
    int num_kv_heads  = 32;      ///< Number of KV heads (for GQA)
    int num_layers    = 32;      ///< Number of transformer layers
    int max_seq_len   = 4096;    ///< Maximum sequence length
    int intermediate  = 11008;   ///< MLP intermediate dimension
    float norm_eps    = 1e-5f;   ///< RMSNorm epsilon
    float rope_theta  = 10000.0f; ///< RoPE base frequency (Llama-2: 10000, Llama-3: 500000)
};

/// @class Model
/// @brief Manages model weights and configuration.
///
/// Loads raw binary weight files produced by tools/convert_model.py
/// and stores them as named Tensors on the GPU.
class Model {
public:
    Model();
    ~Model();

    /// Load model weights from a binary file.
    /// @param path     Path to the binary weight file
    /// @param config   Model configuration
    /// @return true on success, false on failure
    bool load(const std::string& path, const ModelConfig& config);

    /// Retrieve a named weight tensor.
    /// @param name Weight name (e.g., "layers.0.attention.wq")
    /// @return Const reference to the tensor
    const Tensor& get_weight(const std::string& name) const;

    /// Check if the model is loaded.
    bool is_loaded() const;

    /// Get the model configuration.
    const ModelConfig& config() const;

    /// Print summary of loaded weights.
    void print_summary() const;

private:
    ModelConfig                            config_;
    std::unordered_map<std::string, Tensor> weights_;
    bool                                   loaded_;
};

} // namespace vulcan
