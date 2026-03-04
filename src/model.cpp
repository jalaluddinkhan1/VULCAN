/// @file model.cpp
/// @brief VULCAN Model — Weight loading implementation.
///
/// Loads raw binary weight files into GPU tensors. The binary format
/// is produced by tools/convert_model.py from HuggingFace checkpoints.
///
/// Binary format (v1):
///   [4 bytes]  magic number (0x56554C43 = "VULC")
///   [4 bytes]  version
///   [4 bytes]  num_weights
///   For each weight:
///     [4 bytes]  name_length
///     [N bytes]  name (ASCII)
///     [4 bytes]  num_dims
///     [4*D bytes] shape (D integers)
///     [4*numel bytes] float32 data

#include "vulcan/model.h"
#include "cuda/utils.h"
#include <fstream>
#include <iostream>
#include <numeric>

namespace vulcan {

// Magic number: "VULC" in hex
static constexpr uint32_t VULCAN_MAGIC = 0x56554C43;
static constexpr uint32_t VULCAN_VERSION = 1;

Model::Model() : loaded_(false) {}

Model::~Model() = default;

bool Model::load(const std::string& path, const ModelConfig& config) {
    config_ = config;

    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[VULCAN] Error: Cannot open model file: " << path << std::endl;
        return false;
    }

    // Read and validate header
    uint32_t magic, version, num_weights;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&version), 4);
    file.read(reinterpret_cast<char*>(&num_weights), 4);

    if (magic != VULCAN_MAGIC) {
        std::cerr << "[VULCAN] Error: Invalid magic number. Expected VULC format." << std::endl;
        return false;
    }

    if (version != VULCAN_VERSION) {
        std::cerr << "[VULCAN] Error: Unsupported format version " << version << std::endl;
        return false;
    }

    std::cout << "[VULCAN] Loading " << num_weights << " weight tensors..." << std::endl;

    for (uint32_t i = 0; i < num_weights; ++i) {
        // Read weight name
        uint32_t name_len;
        file.read(reinterpret_cast<char*>(&name_len), 4);
        std::string name(name_len, '\0');
        file.read(name.data(), name_len);

        // Read shape
        uint32_t num_dims;
        file.read(reinterpret_cast<char*>(&num_dims), 4);
        std::vector<int> shape(num_dims);
        file.read(reinterpret_cast<char*>(shape.data()), num_dims * 4);

        // Calculate total elements
        size_t numel = 1;
        for (int d : shape) numel *= d;

        // Read float data into CPU buffer
        std::vector<float> cpu_data(numel);
        file.read(reinterpret_cast<char*>(cpu_data.data()), numel * sizeof(float));

        if (!file) {
            std::cerr << "[VULCAN] Error: Unexpected EOF reading weight '" << name << "'" << std::endl;
            return false;
        }

        // Create GPU tensor and transfer
        Tensor tensor(shape, Device::CUDA);
        tensor.from_host(cpu_data.data());

        std::cout << "  [" << (i + 1) << "/" << num_weights << "] "
                  << name << " — shape=[";
        for (size_t d = 0; d < shape.size(); ++d) {
            if (d > 0) std::cout << ", ";
            std::cout << shape[d];
        }
        std::cout << "]" << std::endl;

        weights_.emplace(std::move(name), std::move(tensor));
    }

    loaded_ = true;
    std::cout << "[VULCAN] Model loaded successfully. "
              << weights_.size() << " tensors on GPU." << std::endl;
    return true;
}

const Tensor& Model::get_weight(const std::string& name) const {
    auto it = weights_.find(name);
    if (it == weights_.end()) {
        throw std::runtime_error("[VULCAN] Weight not found: " + name);
    }
    return it->second;
}

bool Model::is_loaded() const {
    return loaded_;
}

const ModelConfig& Model::config() const {
    return config_;
}

void Model::print_summary() const {
    std::cout << "\n╔══════════════════════════════════════════════╗" << std::endl;
    std::cout << "║       VULCAN — Model Summary                 ║" << std::endl;
    std::cout << "╠══════════════════════════════════════════════╣" << std::endl;
    std::cout << "║  Loaded:        " << (loaded_ ? "Yes" : "No") << std::endl;
    std::cout << "║  Vocab Size:    " << config_.vocab_size << std::endl;
    std::cout << "║  Hidden Dim:    " << config_.hidden_dim << std::endl;
    std::cout << "║  Num Heads:     " << config_.num_heads << std::endl;
    std::cout << "║  Num KV Heads:  " << config_.num_kv_heads << std::endl;
    std::cout << "║  Num Layers:    " << config_.num_layers << std::endl;
    std::cout << "║  Max Seq Len:   " << config_.max_seq_len << std::endl;
    std::cout << "║  RoPE Theta:    " << config_.rope_theta << std::endl;
    std::cout << "║  Num Weights:   " << weights_.size() << std::endl;

    size_t total_params = 0;
    for (const auto& [name, tensor] : weights_) {
        total_params += tensor.numel();
    }
    std::cout << "║  Total Params:  " << (total_params / 1e6) << "M" << std::endl;
    std::cout << "╚══════════════════════════════════════════════╝\n" << std::endl;
}

} // namespace vulcan
