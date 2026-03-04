/// @file inspect_weights.cpp
/// @brief CLI tool to inspect VULCAN binary weight files.
///
/// Reads a .vulcan weight file and prints metadata about each tensor:
///   - Name, shape, number of parameters
///   - Data statistics (min, max, mean, std)
///   - Total file size and parameter count
///
/// Usage:
///   ./inspect_weights model.vulcan
///   ./inspect_weights model.vulcan --stats    # Include per-tensor statistics

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <numeric>

static constexpr uint32_t VULCAN_MAGIC = 0x56554C43;

struct TensorInfo {
    std::string name;
    std::vector<int> shape;
    size_t numel;
    float min_val, max_val, mean_val, std_val;
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: inspect_weights <model.vulcan> [--stats]" << std::endl;
        return 1;
    }

    std::string path = argv[1];
    bool show_stats = (argc > 2 && std::string(argv[2]) == "--stats");

    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open: " << path << std::endl;
        return 1;
    }

    // Read header
    uint32_t magic, version, num_weights;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&version), 4);
    file.read(reinterpret_cast<char*>(&num_weights), 4);

    if (magic != VULCAN_MAGIC) {
        std::cerr << "[ERROR] Not a VULCAN weight file (bad magic)." << std::endl;
        return 1;
    }

    std::cout << "\n╔══════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  VULCAN — Weight Inspector                   ║" << std::endl;
    std::cout << "╠══════════════════════════════════════════════╣" << std::endl;
    std::cout << "║  File:     " << path << std::endl;
    std::cout << "║  Version:  " << version << std::endl;
    std::cout << "║  Weights:  " << num_weights << std::endl;
    std::cout << "╚══════════════════════════════════════════════╝\n" << std::endl;

    size_t total_params = 0;

    for (uint32_t i = 0; i < num_weights; ++i) {
        // Read name
        uint32_t name_len;
        file.read(reinterpret_cast<char*>(&name_len), 4);
        std::string name(name_len, '\0');
        file.read(name.data(), name_len);

        // Read shape
        uint32_t num_dims;
        file.read(reinterpret_cast<char*>(&num_dims), 4);
        std::vector<int> shape(num_dims);
        file.read(reinterpret_cast<char*>(shape.data()), num_dims * 4);

        size_t numel = 1;
        for (int d : shape) numel *= d;
        total_params += numel;

        // Print info
        std::cout << "  [" << (i + 1) << "/" << num_weights << "] "
                  << name << "  shape=[";
        for (size_t d = 0; d < shape.size(); ++d) {
            if (d > 0) std::cout << ", ";
            std::cout << shape[d];
        }
        std::cout << "]  params=" << numel;

        if (show_stats) {
            // Read and compute statistics
            std::vector<float> data(numel);
            file.read(reinterpret_cast<char*>(data.data()), numel * sizeof(float));

            float min_v = *std::min_element(data.begin(), data.end());
            float max_v = *std::max_element(data.begin(), data.end());
            float mean = std::accumulate(data.begin(), data.end(), 0.0f) / numel;
            float var = 0.0f;
            for (float x : data) var += (x - mean) * (x - mean);
            float std_v = std::sqrt(var / numel);

            std::cout << "  min=" << min_v << " max=" << max_v
                      << " mean=" << mean << " std=" << std_v;
        } else {
            // Skip data
            file.seekg(numel * sizeof(float), std::ios::cur);
        }

        std::cout << std::endl;
    }

    std::cout << "\n  Total parameters: " << total_params
              << " (" << (total_params / 1e6) << "M)" << std::endl;

    // File size
    file.seekg(0, std::ios::end);
    auto file_size = file.tellg();
    std::cout << "  File size: " << (file_size / (1024.0 * 1024.0))
              << " MB\n" << std::endl;

    return 0;
}
