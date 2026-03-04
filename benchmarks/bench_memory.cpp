/// @file bench_memory.cpp
/// @brief VULCAN Memory Benchmark — Paged allocator and KV cache profiling.
///
/// Measures:
///   - Sequential allocation latency
///   - Random free/realloc fragmentation behavior
///   - GPU ↔ CPU swap round-trip latency
///   - KV cache allocation and memory footprint
///   - Comparison: paged vs linear allocation


#include "cuda/memory.h"
#include "cuda/utils.h"
#include "vulcan/kv_cache.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <algorithm>
#include <cstring>

using Clock = std::chrono::high_resolution_clock;

int main() {
    std::cout << "╔══════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  VULCAN — Memory Benchmark                 ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════╝\n" << std::endl;

    vulcan::cuda::print_device_info();
    std::cout << std::endl;

    // Print GPU memory baseline
    size_t free_bytes, total_bytes;
    vulcan::cuda::get_memory_info(free_bytes, total_bytes);
    printf("  Total VRAM: %.1f MB\n", total_bytes / (1024.0 * 1024.0));
    printf("  Free VRAM:  %.1f MB\n\n", free_bytes / (1024.0 * 1024.0));

    // ── 1. GPUBuffer Allocation Benchmark ───────────────────────────
    std::cout << "═══ GPUBuffer Allocation ═══\n" << std::endl;
    {
        const int num_allocs = 1000;
        const size_t alloc_size = 16 * 1024;  // 16KB

        auto start = Clock::now();
        std::vector<vulcan::cuda::GPUBuffer> buffers;
        buffers.reserve(num_allocs);
        for (int i = 0; i < num_allocs; ++i) {
            buffers.emplace_back(alloc_size);
        }
        cudaDeviceSynchronize();
        auto mid = Clock::now();

        // Free all
        buffers.clear();
        cudaDeviceSynchronize();
        auto end = Clock::now();

        double alloc_us = std::chrono::duration<double, std::micro>(mid - start).count();
        double free_us = std::chrono::duration<double, std::micro>(end - mid).count();

        printf("  %d allocations × %zu KB:\n", num_allocs, alloc_size / 1024);
        printf("    Alloc total:  %.1f us (%.1f us/alloc)\n",
               alloc_us, alloc_us / num_allocs);
        printf("    Free total:   %.1f us (%.1f us/free)\n\n",
               free_us, free_us / num_allocs);
    }

    // ── 2. Paged Allocator Benchmark ────────────────────────────────
    std::cout << "═══ Paged Allocator (MemoryManager) ═══\n" << std::endl;
    {
        const size_t page_size = 16 * 1024;
        const int max_pages = 256;

        auto start = Clock::now();
        vulcan::cuda::MemoryManager mgr(page_size, max_pages);
        auto init_end = Clock::now();

        double init_us = std::chrono::duration<double, std::micro>(init_end - start).count();
        printf("  Pool init (%d pages × %zu KB): %.1f us\n\n",
               max_pages, page_size / 1024, init_us);

        // Sequential allocation
        start = Clock::now();
        std::vector<int> ids;
        ids.reserve(max_pages);
        for (int i = 0; i < max_pages; ++i) {
            ids.push_back(mgr.allocate_block());
        }
        auto alloc_end = Clock::now();
        double alloc_us = std::chrono::duration<double, std::micro>(alloc_end - start).count();
        printf("  Sequential alloc %d blocks: %.1f us (%.2f us/block)\n",
               max_pages, alloc_us, alloc_us / max_pages);
        printf("  Blocks in use:    %d\n", mgr.blocks_in_use());
        printf("  Blocks available: %d\n", mgr.blocks_available());
        printf("  Total allocated:  %.1f MB\n\n",
               mgr.total_allocated_bytes() / (1024.0 * 1024.0));

        // Random free/realloc (fragmentation test)
        std::mt19937 rng(42);
        std::shuffle(ids.begin(), ids.end(), rng);

        // Free half randomly
        start = Clock::now();
        for (int i = 0; i < max_pages / 2; ++i) {
            mgr.free_block(ids[i]);
        }
        auto free_end = Clock::now();
        double free_us = std::chrono::duration<double, std::micro>(free_end - start).count();
        printf("  Random free %d blocks: %.1f us\n", max_pages / 2, free_us);

        // Reallocate
        start = Clock::now();
        for (int i = 0; i < max_pages / 2; ++i) {
            mgr.allocate_block();
        }
        auto realloc_end = Clock::now();
        double realloc_us = std::chrono::duration<double, std::micro>(realloc_end - start).count();
        printf("  Realloc %d blocks:     %.1f us\n", max_pages / 2, realloc_us);
        printf("  (Zero fragmentation by design — paged allocator)\n\n");

        // Swap benchmark
        start = Clock::now();
        bool swapped = mgr.swap_to_cpu(ids[max_pages / 2]);
        auto swap_out = Clock::now();
        double swap_out_us = std::chrono::duration<double, std::micro>(swap_out - start).count();

        bool restored = mgr.swap_to_gpu(ids[max_pages / 2]);
        auto swap_in = Clock::now();
        double swap_in_us = std::chrono::duration<double, std::micro>(swap_in - swap_out).count();

        printf("  GPU → CPU swap (%zu KB): %.1f us %s\n",
               page_size / 1024, swap_out_us, swapped ? "✓" : "✗");
        printf("  CPU → GPU swap (%zu KB): %.1f us %s\n\n",
               page_size / 1024, swap_in_us, restored ? "✓" : "✗");
    }

    // ── 3. KV Cache Benchmark ───────────────────────────────────────
    std::cout << "═══ KV Cache Allocation ═══\n" << std::endl;
    {
        // Llama-2-7B dimensions
        struct Config {
            const char* name;
            int kv_heads, max_seq, head_dim, layers;
        };
        Config configs[] = {
            {"Llama-2-7B (4K ctx)",   32, 4096,  128, 32},
            {"Llama-2-7B (8K ctx)",   32, 8192,  128, 32},
            {"Llama-2-13B (4K ctx)",  40, 4096,  128, 40},
        };

        for (auto& cfg : configs) {
            auto start = Clock::now();
            vulcan::KVCache cache(cfg.kv_heads, cfg.max_seq, cfg.head_dim, cfg.layers);
            auto end = Clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count();

            printf("  %-24s  %7.1f MB  alloc: %.1f ms\n",
                   cfg.name,
                   cache.memory_usage() / (1024.0 * 1024.0),
                   ms);
        }
    }

    // ── 4. Final Memory State ───────────────────────────────────────
    std::cout << std::endl;
    vulcan::cuda::get_memory_info(free_bytes, total_bytes);
    size_t used = total_bytes - free_bytes;
    printf("  Final VRAM: %.1f MB used / %.1f MB total (%.1f%% utilization)\n",
           used / (1024.0 * 1024.0),
           total_bytes / (1024.0 * 1024.0),
           100.0 * used / total_bytes);

    std::cout << "\n[BENCH] Done." << std::endl;
    return 0;
}
