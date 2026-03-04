#pragma once

/// @file utils.h
/// @brief CUDA utility macros and helpers.
///
/// Error checking macros that should wrap EVERY CUDA API call.
/// Using these ensures no silent failures during development.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace vulcan {
namespace cuda {

// ─── Error Checking Macros ──────────────────────────────────────────────────

/// Check the return value of a CUDA API call.
/// Usage: CUDA_CHECK(cudaMalloc(&ptr, size));
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error at %s:%d — %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            fprintf(stderr, "  Call: %s\n", #call);                           \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

/// Check for errors after a kernel launch (catches async errors).
/// Usage: my_kernel<<<grid, block>>>(...); CUDA_CHECK_LAST();
#define CUDA_CHECK_LAST()                                                     \
    do {                                                                      \
        cudaError_t err = cudaGetLastError();                                 \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Kernel Error at %s:%d — %s\n",             \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

/// Synchronize device and check for errors.
/// Usage: CUDA_SYNC_CHECK();
#define CUDA_SYNC_CHECK()                                                     \
    do {                                                                      \
        cudaError_t err = cudaDeviceSynchronize();                            \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Sync Error at %s:%d — %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// ─── Device Helpers ─────────────────────────────────────────────────────────

/// Print GPU device information.
inline void print_device_info() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));

    printf("╔══════════════════════════════════════════════╗\n");
    printf("║  VULCAN — GPU Device Info                    ║\n");
    printf("╠══════════════════════════════════════════════╣\n");
    printf("║  Device:          %s\n", props.name);
    printf("║  Compute:         %d.%d\n", props.major, props.minor);
    printf("║  VRAM:            %.1f GB\n",
           props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("║  SM Count:        %d\n", props.multiProcessorCount);
    printf("║  Max Threads/SM:  %d\n", props.maxThreadsPerMultiProcessor);
    printf("║  Shared Mem/SM:   %.1f KB\n",
           props.sharedMemPerMultiprocessor / 1024.0);
    printf("║  Warp Size:       %d\n", props.warpSize);
    printf("╚══════════════════════════════════════════════╝\n");
}

/// Get available and total GPU memory in bytes.
inline void get_memory_info(size_t& free_bytes, size_t& total_bytes) {
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
}

// ─── Kernel Launch Helpers ──────────────────────────────────────────────────

/// Calculate grid dimensions for 1D kernel launch.
/// @param total_threads Total number of threads needed
/// @param block_size    Threads per block (default: 256)
/// @return Number of blocks
inline int calc_grid_1d(int total_threads, int block_size = 256) {
    return (total_threads + block_size - 1) / block_size;
}

} // namespace cuda
} // namespace vulcan
