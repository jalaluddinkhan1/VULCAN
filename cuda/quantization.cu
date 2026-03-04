/// @file quantization.cu
/// @brief INT4 Dequantization — CUDA kernel.
///
/// Converts packed 4-bit integer weights to float32 for computation.
/// Quantization format: group-wise with per-group scale factors.
///
/// Packing: Two INT4 values packed per uint8_t byte.
///   Byte: [high_nibble | low_nibble]
///   low_nibble  = byte & 0x0F       → values 0–15 (mapped to -8 to +7)
///   high_nibble = (byte >> 4) & 0x0F → values 0–15 (mapped to -8 to +7)
///
/// Dequantization:
///   float_val = (int4_val - 8) * scale[group_idx]


#include "cuda/kernels.h"
#include "cuda/utils.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace vulcan {
namespace cuda {

// ─── Constants ──────────────────────────────────────────────────────────────

constexpr int DEQUANT_BLOCK_SIZE = 256;

// ─── CUDA Kernel ────────────────────────────────────────────────────────────

/// INT4 dequantization kernel.
///
/// Each thread processes 2 output elements (one packed byte).
///
/// @param input       Packed INT4 weights (n/2 bytes)
/// @param scales      Per-group scale factors (n / group_size floats)
/// @param output      Dequantized float output (n floats)
/// @param n           Number of output elements
/// @param group_size  Quantization group size (e.g., 128)
__global__ void dequantize_int4_kernel(const uint8_t* __restrict__ input,
                                       const float* __restrict__ scales,
                                       float* __restrict__ output,
                                       int n, int group_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int byte_idx = idx;  // Each thread processes one byte → two elements
    int out_idx = byte_idx * 2;

    if (out_idx + 1 >= n) return;

    uint8_t packed = input[byte_idx];

    // Extract two 4-bit values
    int low  = static_cast<int>(packed & 0x0F) - 8;   // Map [0,15] → [-8,7]
    int high = static_cast<int>(packed >> 4) - 8;

    // Look up group scale
    int group_low  = out_idx / group_size;
    int group_high = (out_idx + 1) / group_size;

    // Dequantize
    output[out_idx]     = static_cast<float>(low) * scales[group_low];
    output[out_idx + 1] = static_cast<float>(high) * scales[group_high];
}

// ─── Host Wrapper ───────────────────────────────────────────────────────────

void launch_dequantize_int4(const uint8_t* input, const float* scales,
                            float* output, int n, int group_size) {
    int num_bytes = (n + 1) / 2;  // ceil(n / 2)
    const int block_size = DEQUANT_BLOCK_SIZE;
    const int grid_size = (num_bytes + block_size - 1) / block_size;

    dequantize_int4_kernel<<<grid_size, block_size>>>(
        input, scales, output, n, group_size
    );

    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();
}

} // namespace cuda
} // namespace vulcan
