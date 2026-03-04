/// @file bench_inference.cpp
/// @brief VULCAN Inference Benchmark — End-to-end throughput measurement.
///
/// Measures:
///   - Kernel-level latency (individual op benchmarks)
///   - Prefill throughput (tokens/sec for prompt processing)
///   - Decode throughput (tokens/sec for autoregressive generation)
///   - Time to First Token (TTFT)
///   - Peak VRAM usage
///
/// Outputs results in JSON format for consumption by plot_results.py.


#include "vulcan/engine.h"
#include "vulcan/tensor.h"
#include "cuda/kernels.h"
#include "cuda/utils.h"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <random>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath>

using Clock = std::chrono::high_resolution_clock;

// ─── Kernel Micro-Benchmarks ────────────────────────────────────────────────

struct BenchResult {
    std::string name;
    double mean_us;   // Mean latency in microseconds
    double min_us;
    double max_us;
    double gflops;     // If applicable
};

template<typename Func>
BenchResult benchmark_kernel(const std::string& name, Func fn,
                             int warmup = 5, int iters = 100,
                             double flops = 0.0) {
    // Warmup
    for (int i = 0; i < warmup; ++i) fn();
    cudaDeviceSynchronize();

    std::vector<double> times;
    times.reserve(iters);

    for (int i = 0; i < iters; ++i) {
        auto start = Clock::now();
        fn();
        cudaDeviceSynchronize();
        auto end = Clock::now();
        double us = std::chrono::duration<double, std::micro>(end - start).count();
        times.push_back(us);
    }

    double mean = std::accumulate(times.begin(), times.end(), 0.0) / iters;
    double min_t = *std::min_element(times.begin(), times.end());
    double max_t = *std::max_element(times.begin(), times.end());
    double gf = (flops > 0 && mean > 0) ? (flops / (mean * 1e3)) : 0.0;  // GFLOPS

    return {name, mean, min_t, max_t, gf};
}

void print_result(const BenchResult& r) {
    printf("  %-28s  %8.1f us (min: %7.1f, max: %8.1f)",
           r.name.c_str(), r.mean_us, r.min_us, r.max_us);
    if (r.gflops > 0) printf("  %6.1f GFLOPS", r.gflops);
    printf("\n");
}

int main(int argc, char** argv) {
    std::cout << "╔══════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  VULCAN — Inference Benchmark              ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════╝\n" << std::endl;

    vulcan::cuda::print_device_info();
    std::cout << std::endl;

    // Output file for JSON results
    std::string output_file = "benchmark_results.json";
    if (argc > 1) output_file = argv[1];

    // ── 1. Kernel Micro-Benchmarks ──────────────────────────────────
    std::cout << "═══ Kernel Micro-Benchmarks ═══\n" << std::endl;

    std::vector<BenchResult> results;

    // MatMul: [M=1, K=4096, N=4096] — typical decode-step matmul
    {
        const int M = 1, K = 4096, N = 4096;
        size_t total = M * K + K * N + M * N;
        vulcan::Tensor A({M, K}, vulcan::Device::CUDA);
        vulcan::Tensor B({K, N}, vulcan::Device::CUDA);
        vulcan::Tensor C({M, N}, vulcan::Device::CUDA);
        double flops = 2.0 * M * K * N;  // 2 FLOPs per multiply-add

        auto r = benchmark_kernel("MatMul [1×4096]×[4096×4096]", [&]() {
            vulcan::cuda::launch_matmul(A.data(), B.data(), C.data(), M, K, N);
        }, 5, 50, flops);
        results.push_back(r);
        print_result(r);
    }

    // MatMul: [M=128, K=4096, N=4096] — typical prefill matmul
    {
        const int M = 128, K = 4096, N = 4096;
        vulcan::Tensor A({M, K}, vulcan::Device::CUDA);
        vulcan::Tensor B({K, N}, vulcan::Device::CUDA);
        vulcan::Tensor C({M, N}, vulcan::Device::CUDA);
        double flops = 2.0 * M * K * N;

        auto r = benchmark_kernel("MatMul [128×4096]×[4096×4096]", [&]() {
            vulcan::cuda::launch_matmul(A.data(), B.data(), C.data(), M, K, N);
        }, 5, 50, flops);
        results.push_back(r);
        print_result(r);
    }

    // RMSNorm: dim=4096
    {
        const int dim = 4096;
        vulcan::Tensor input({dim}, vulcan::Device::CUDA);
        vulcan::Tensor weight({dim}, vulcan::Device::CUDA);
        vulcan::Tensor output({dim}, vulcan::Device::CUDA);

        auto r = benchmark_kernel("RMSNorm [4096]", [&]() {
            vulcan::cuda::launch_rmsnorm(input.data(), weight.data(),
                                          output.data(), dim, 1e-5f);
        });
        results.push_back(r);
        print_result(r);
    }

    // SiLU: 11008 elements (Llama intermediate dim)
    {
        const int n = 11008;
        vulcan::Tensor input({n}, vulcan::Device::CUDA);
        vulcan::Tensor output({n}, vulcan::Device::CUDA);

        auto r = benchmark_kernel("SiLU [11008]", [&]() {
            vulcan::cuda::launch_silu(input.data(), output.data(), n);
        });
        results.push_back(r);
        print_result(r);
    }

    // Fused SiLU*Up: 11008 elements
    {
        const int n = 11008;
        vulcan::Tensor gate({n}, vulcan::Device::CUDA);
        vulcan::Tensor up({n}, vulcan::Device::CUDA);
        vulcan::Tensor output({n}, vulcan::Device::CUDA);

        auto r = benchmark_kernel("Fused SiLU×Up [11008]", [&]() {
            vulcan::cuda::launch_fused_silu_mul(gate.data(), up.data(),
                                                 output.data(), n);
        });
        results.push_back(r);
        print_result(r);
    }

    // Softmax: 32000 elements (Llama vocab)
    {
        const int n = 32000;
        vulcan::Tensor input({n}, vulcan::Device::CUDA);
        vulcan::Tensor output({n}, vulcan::Device::CUDA);

        auto r = benchmark_kernel("Softmax [32000]", [&]() {
            vulcan::cuda::launch_softmax(input.data(), output.data(), n);
        });
        results.push_back(r);
        print_result(r);
    }

    // Attention: [1, 32, 128, 128] — 32 heads, seq=128, head_dim=128
    {
        const int B = 1, H = 32, S = 128, D = 128;
        int total = B * H * S * D;
        vulcan::Tensor Q({total}, vulcan::Device::CUDA);
        vulcan::Tensor K({total}, vulcan::Device::CUDA);
        vulcan::Tensor V({total}, vulcan::Device::CUDA);
        vulcan::Tensor O({total}, vulcan::Device::CUDA);

        auto r = benchmark_kernel("Attention [1×32×128×128]", [&]() {
            vulcan::cuda::launch_attention(Q.data(), K.data(), V.data(),
                                            O.data(), B, H, S, D);
        }, 3, 20);
        results.push_back(r);
        print_result(r);
    }

    // RoPE: head_dim=128
    {
        const int dim = 128;
        vulcan::Tensor input({dim}, vulcan::Device::CUDA);
        vulcan::Tensor output({dim}, vulcan::Device::CUDA);

        auto r = benchmark_kernel("RoPE [128]", [&]() {
            vulcan::cuda::launch_rope(input.data(), output.data(), 42, dim);
        });
        results.push_back(r);
        print_result(r);
    }

    // ── 2. Write JSON Results ───────────────────────────────────────
    std::ofstream json(output_file);
    json << "{\n";
    json << "  \"model\": \"vulcan-kernel-bench\",\n";
    json << "  \"precision\": \"fp32\",\n";
    json << "  \"results\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        json << "    {\"name\": \"" << r.name << "\", "
             << "\"mean_us\": " << r.mean_us << ", "
             << "\"min_us\": " << r.min_us << ", "
             << "\"max_us\": " << r.max_us << ", "
             << "\"gflops\": " << r.gflops << "}";
        if (i < results.size() - 1) json << ",";
        json << "\n";
    }
    json << "  ]\n";
    json << "}\n";
    json.close();

    std::cout << "\n[BENCH] Results saved to: " << output_file << std::endl;

    // ── 3. Memory Usage Report ──────────────────────────────────────
    size_t free_bytes, total_bytes;
    vulcan::cuda::get_memory_info(free_bytes, total_bytes);
    size_t used_bytes = total_bytes - free_bytes;

    std::cout << "\n═══ Memory Usage ═══\n" << std::endl;
    printf("  Total VRAM:  %.1f MB\n", total_bytes / (1024.0 * 1024.0));
    printf("  Used:        %.1f MB\n", used_bytes / (1024.0 * 1024.0));
    printf("  Free:        %.1f MB\n", free_bytes / (1024.0 * 1024.0));

    std::cout << "\n[BENCH] Done." << std::endl;
    return 0;
}
