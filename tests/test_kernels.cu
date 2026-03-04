/// @file test_kernels.cu
/// @brief Golden tests for all CUDA kernels against CPU references.
///
/// Verification strategy (Staff-level rigor):
///   1. Generate random inputs with fixed seed (reproducible)
///   2. Compute reference output on CPU
///   3. Run VULCAN CUDA kernel on GPU
///   4. Compare: max(abs(ref - vulcan)) < tolerance
///
/// Tolerances:
///   - FP32 element-wise ops: 1e-5
///   - FP32 matmul (accumulated): 1e-3 (relaxed due to FP accumulation order)
///   - FP32 RMSNorm: 1e-4

#include <gtest/gtest.h>
#include "cuda/kernels.h"
#include "cuda/utils.h"
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

// ═══════════════════════════════════════════════════════════════════════════
//  Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Generate a vector of random floats in [lo, hi] with a given seed.
static std::vector<float> random_floats(int n, float lo = -1.0f,
                                        float hi = 1.0f, int seed = 123) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) v[i] = dist(rng);
    return v;
}

/// Compute max absolute error between two vectors.
static float max_abs_error(const std::vector<float>& a,
                           const std::vector<float>& b) {
    float max_err = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        max_err = std::max(max_err, std::abs(a[i] - b[i]));
    }
    return max_err;
}

/// RAII wrapper for GPU test buffers — prevents leaks on test failure.
struct GPUMem {
    float* ptr = nullptr;
    GPUMem(size_t n) { CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(float))); }
    ~GPUMem() { if (ptr) cudaFree(ptr); }
    void upload(const float* host, size_t n) {
        CUDA_CHECK(cudaMemcpy(ptr, host, n * sizeof(float), cudaMemcpyHostToDevice));
    }
    void download(float* host, size_t n) {
        CUDA_CHECK(cudaMemcpy(host, ptr, n * sizeof(float), cudaMemcpyDeviceToHost));
    }
};

// ═══════════════════════════════════════════════════════════════════════════
//  CPU Reference Implementations
// ═══════════════════════════════════════════════════════════════════════════

/// CPU reference: naive matrix multiply C = A * B
static std::vector<float> cpu_matmul(const std::vector<float>& A,
                                     const std::vector<float>& B,
                                     int M, int K, int N) {
    std::vector<float> C(M * N, 0.0f);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < K; ++k)
                C[i * N + j] += A[i * K + k] * B[k * N + j];
    return C;
}

/// CPU reference: SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
static float cpu_silu(float x) {
    return x / (1.0f + std::exp(-x));
}

/// CPU reference: ReLU(x) = max(0, x)
static float cpu_relu(float x) {
    return std::max(0.0f, x);
}

/// CPU reference: RMSNorm
static std::vector<float> cpu_rmsnorm(const std::vector<float>& input,
                                      const std::vector<float>& weight,
                                      float eps) {
    int n = static_cast<int>(input.size());
    float ss = 0.0f;
    for (float x : input) ss += x * x;
    float rms = 1.0f / std::sqrt(ss / n + eps);

    std::vector<float> output(n);
    for (int i = 0; i < n; ++i) {
        output[i] = input[i] * rms * weight[i];
    }
    return output;
}

// ═══════════════════════════════════════════════════════════════════════════
//  MatMul Tests
// ═══════════════════════════════════════════════════════════════════════════

TEST(MatMulTest, Square_4x4) {
    const int M = 4, K = 4, N = 4;
    auto h_A = random_floats(M * K, -1.0f, 1.0f, 42);
    auto h_B = random_floats(K * N, -1.0f, 1.0f, 43);
    auto ref = cpu_matmul(h_A, h_B, M, K, N);

    GPUMem d_A(M * K), d_B(K * N), d_C(M * N);
    d_A.upload(h_A.data(), M * K);
    d_B.upload(h_B.data(), K * N);

    vulcan::cuda::launch_matmul(d_A.ptr, d_B.ptr, d_C.ptr, M, K, N);

    std::vector<float> h_C(M * N);
    d_C.download(h_C.data(), M * N);

    float err = max_abs_error(h_C, ref);
    EXPECT_LT(err, 1e-4f) << "Max absolute error: " << err;
}

TEST(MatMulTest, Square_64x64) {
    const int M = 64, K = 64, N = 64;
    auto h_A = random_floats(M * K, -0.5f, 0.5f, 100);
    auto h_B = random_floats(K * N, -0.5f, 0.5f, 101);
    auto ref = cpu_matmul(h_A, h_B, M, K, N);

    GPUMem d_A(M * K), d_B(K * N), d_C(M * N);
    d_A.upload(h_A.data(), M * K);
    d_B.upload(h_B.data(), K * N);

    vulcan::cuda::launch_matmul(d_A.ptr, d_B.ptr, d_C.ptr, M, K, N);

    std::vector<float> h_C(M * N);
    d_C.download(h_C.data(), M * N);

    float err = max_abs_error(h_C, ref);
    EXPECT_LT(err, 1e-3f) << "Max absolute error: " << err;
}

TEST(MatMulTest, NonSquare_128x64x32) {
    const int M = 128, K = 64, N = 32;
    auto h_A = random_floats(M * K, -0.5f, 0.5f, 200);
    auto h_B = random_floats(K * N, -0.5f, 0.5f, 201);
    auto ref = cpu_matmul(h_A, h_B, M, K, N);

    GPUMem d_A(M * K), d_B(K * N), d_C(M * N);
    d_A.upload(h_A.data(), M * K);
    d_B.upload(h_B.data(), K * N);

    vulcan::cuda::launch_matmul(d_A.ptr, d_B.ptr, d_C.ptr, M, K, N);

    std::vector<float> h_C(M * N);
    d_C.download(h_C.data(), M * N);

    float err = max_abs_error(h_C, ref);
    EXPECT_LT(err, 1e-3f) << "Max absolute error: " << err;
}

TEST(MatMulTest, NonAligned_33x17x45) {
    // Dimensions that are NOT multiples of TILE_SIZE (32)
    const int M = 33, K = 17, N = 45;
    auto h_A = random_floats(M * K, -1.0f, 1.0f, 300);
    auto h_B = random_floats(K * N, -1.0f, 1.0f, 301);
    auto ref = cpu_matmul(h_A, h_B, M, K, N);

    GPUMem d_A(M * K), d_B(K * N), d_C(M * N);
    d_A.upload(h_A.data(), M * K);
    d_B.upload(h_B.data(), K * N);

    vulcan::cuda::launch_matmul(d_A.ptr, d_B.ptr, d_C.ptr, M, K, N);

    std::vector<float> h_C(M * N);
    d_C.download(h_C.data(), M * N);

    float err = max_abs_error(h_C, ref);
    EXPECT_LT(err, 1e-3f) << "Max absolute error: " << err;
}

TEST(MatMulTest, LargeMatrix_256x512x256) {
    const int M = 256, K = 512, N = 256;
    auto h_A = random_floats(M * K, -0.1f, 0.1f, 400);
    auto h_B = random_floats(K * N, -0.1f, 0.1f, 401);
    auto ref = cpu_matmul(h_A, h_B, M, K, N);

    GPUMem d_A(M * K), d_B(K * N), d_C(M * N);
    d_A.upload(h_A.data(), M * K);
    d_B.upload(h_B.data(), K * N);

    vulcan::cuda::launch_matmul(d_A.ptr, d_B.ptr, d_C.ptr, M, K, N);

    std::vector<float> h_C(M * N);
    d_C.download(h_C.data(), M * N);

    float err = max_abs_error(h_C, ref);
    // Relaxed tolerance for larger matrices — more FP accumulation error
    EXPECT_LT(err, 5e-3f) << "Max absolute error: " << err;
}

TEST(MatMulTest, IdentityMultiply) {
    // A * I = A — test correctness with known output
    const int N = 32;
    auto h_A = random_floats(N * N, -2.0f, 2.0f, 500);
    std::vector<float> h_I(N * N, 0.0f);
    for (int i = 0; i < N; ++i) h_I[i * N + i] = 1.0f;

    GPUMem d_A(N * N), d_I(N * N), d_C(N * N);
    d_A.upload(h_A.data(), N * N);
    d_I.upload(h_I.data(), N * N);

    vulcan::cuda::launch_matmul(d_A.ptr, d_I.ptr, d_C.ptr, N, N, N);

    std::vector<float> h_C(N * N);
    d_C.download(h_C.data(), N * N);

    float err = max_abs_error(h_C, h_A);
    EXPECT_LT(err, 1e-5f) << "A * I should equal A. Max error: " << err;
}

TEST(MatMulTest, SingleElement) {
    const int M = 1, K = 1, N = 1;
    std::vector<float> h_A = {3.0f};
    std::vector<float> h_B = {4.0f};

    GPUMem d_A(1), d_B(1), d_C(1);
    d_A.upload(h_A.data(), 1);
    d_B.upload(h_B.data(), 1);

    vulcan::cuda::launch_matmul(d_A.ptr, d_B.ptr, d_C.ptr, M, K, N);

    float h_C;
    d_C.download(&h_C, 1);

    EXPECT_NEAR(h_C, 12.0f, 1e-5f);
}

// ═══════════════════════════════════════════════════════════════════════════
//  SiLU (Swish) Tests
// ═══════════════════════════════════════════════════════════════════════════

TEST(SiLUTest, KnownValues) {
    // silu(0) = 0, silu(large) ≈ large, silu(-large) ≈ 0
    const int n = 5;
    std::vector<float> h_in = {0.0f, 1.0f, -1.0f, 5.0f, -5.0f};
    std::vector<float> ref(n);
    for (int i = 0; i < n; ++i) ref[i] = cpu_silu(h_in[i]);

    GPUMem d_in(n), d_out(n);
    d_in.upload(h_in.data(), n);

    vulcan::cuda::launch_silu(d_in.ptr, d_out.ptr, n);

    std::vector<float> h_out(n);
    d_out.download(h_out.data(), n);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(h_out[i], ref[i], 1e-5f)
            << "Mismatch at index " << i << " input=" << h_in[i];
    }
}

TEST(SiLUTest, RandomLargeVector) {
    const int n = 1 << 18;  // 256K elements
    auto h_in = random_floats(n, -10.0f, 10.0f, 600);
    std::vector<float> ref(n);
    for (int i = 0; i < n; ++i) ref[i] = cpu_silu(h_in[i]);

    GPUMem d_in(n), d_out(n);
    d_in.upload(h_in.data(), n);

    vulcan::cuda::launch_silu(d_in.ptr, d_out.ptr, n);

    std::vector<float> h_out(n);
    d_out.download(h_out.data(), n);

    float err = max_abs_error(h_out, ref);
    EXPECT_LT(err, 1e-5f) << "Max SiLU error: " << err;
}

TEST(SiLUTest, ZeroVector) {
    const int n = 1024;
    std::vector<float> h_in(n, 0.0f);

    GPUMem d_in(n), d_out(n);
    d_in.upload(h_in.data(), n);

    vulcan::cuda::launch_silu(d_in.ptr, d_out.ptr, n);

    std::vector<float> h_out(n);
    d_out.download(h_out.data(), n);

    for (int i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(h_out[i], 0.0f);
    }
}

TEST(SiLUTest, NegativeInputs) {
    // SiLU of negative values should be negative (but small magnitude)
    const int n = 4;
    std::vector<float> h_in = {-1.0f, -2.0f, -3.0f, -10.0f};

    GPUMem d_in(n), d_out(n);
    d_in.upload(h_in.data(), n);

    vulcan::cuda::launch_silu(d_in.ptr, d_out.ptr, n);

    std::vector<float> h_out(n);
    d_out.download(h_out.data(), n);

    for (int i = 0; i < n; ++i) {
        float expected = cpu_silu(h_in[i]);
        EXPECT_NEAR(h_out[i], expected, 1e-5f);
        EXPECT_LT(h_out[i], 0.0f) << "SiLU of negative input should be negative";
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  ReLU Tests
// ═══════════════════════════════════════════════════════════════════════════

TEST(ReLUTest, KnownValues) {
    const int n = 6;
    std::vector<float> h_in = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 0.5f};
    std::vector<float> ref =  { 0.0f,  0.0f, 0.0f, 1.0f, 2.0f, 0.5f};

    GPUMem d_in(n), d_out(n);
    d_in.upload(h_in.data(), n);

    vulcan::cuda::launch_relu(d_in.ptr, d_out.ptr, n);

    std::vector<float> h_out(n);
    d_out.download(h_out.data(), n);

    for (int i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(h_out[i], ref[i])
            << "Mismatch at index " << i << " input=" << h_in[i];
    }
}

TEST(ReLUTest, RandomLargeVector) {
    const int n = 1 << 18;
    auto h_in = random_floats(n, -5.0f, 5.0f, 700);
    std::vector<float> ref(n);
    for (int i = 0; i < n; ++i) ref[i] = cpu_relu(h_in[i]);

    GPUMem d_in(n), d_out(n);
    d_in.upload(h_in.data(), n);

    vulcan::cuda::launch_relu(d_in.ptr, d_out.ptr, n);

    std::vector<float> h_out(n);
    d_out.download(h_out.data(), n);

    float err = max_abs_error(h_out, ref);
    EXPECT_LT(err, 1e-6f) << "Max ReLU error: " << err;
}

TEST(ReLUTest, AllNegative) {
    const int n = 256;
    auto h_in = random_floats(n, -10.0f, -0.001f, 710);

    GPUMem d_in(n), d_out(n);
    d_in.upload(h_in.data(), n);

    vulcan::cuda::launch_relu(d_in.ptr, d_out.ptr, n);

    std::vector<float> h_out(n);
    d_out.download(h_out.data(), n);

    for (int i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(h_out[i], 0.0f);
    }
}

TEST(ReLUTest, AllPositive) {
    const int n = 256;
    auto h_in = random_floats(n, 0.001f, 10.0f, 720);

    GPUMem d_in(n), d_out(n);
    d_in.upload(h_in.data(), n);

    vulcan::cuda::launch_relu(d_in.ptr, d_out.ptr, n);

    std::vector<float> h_out(n);
    d_out.download(h_out.data(), n);

    float err = max_abs_error(h_out, h_in);
    EXPECT_LT(err, 1e-6f) << "ReLU of positive inputs should be identity";
}

// ═══════════════════════════════════════════════════════════════════════════
//  RMSNorm Tests
// ═══════════════════════════════════════════════════════════════════════════

TEST(RMSNormTest, UnitWeight) {
    // With weight = all 1s, RMSNorm just normalizes the RMS
    const int n = 128;
    auto h_input = random_floats(n, -1.0f, 1.0f, 800);
    std::vector<float> h_weight(n, 1.0f);
    float eps = 1e-5f;

    auto ref = cpu_rmsnorm(h_input, h_weight, eps);

    GPUMem d_in(n), d_w(n), d_out(n);
    d_in.upload(h_input.data(), n);
    d_w.upload(h_weight.data(), n);

    vulcan::cuda::launch_rmsnorm(d_in.ptr, d_w.ptr, d_out.ptr, n, eps);

    std::vector<float> h_out(n);
    d_out.download(h_out.data(), n);

    float err = max_abs_error(h_out, ref);
    EXPECT_LT(err, 1e-4f) << "Max RMSNorm error: " << err;
}

TEST(RMSNormTest, RandomWeight) {
    const int n = 256;
    auto h_input = random_floats(n, -2.0f, 2.0f, 810);
    auto h_weight = random_floats(n, 0.5f, 2.0f, 811);
    float eps = 1e-5f;

    auto ref = cpu_rmsnorm(h_input, h_weight, eps);

    GPUMem d_in(n), d_w(n), d_out(n);
    d_in.upload(h_input.data(), n);
    d_w.upload(h_weight.data(), n);

    vulcan::cuda::launch_rmsnorm(d_in.ptr, d_w.ptr, d_out.ptr, n, eps);

    std::vector<float> h_out(n);
    d_out.download(h_out.data(), n);

    float err = max_abs_error(h_out, ref);
    EXPECT_LT(err, 1e-4f) << "Max RMSNorm error: " << err;
}

TEST(RMSNormTest, LlamaDimension_4096) {
    // Llama-2-7B uses hidden_dim = 4096
    const int n = 4096;
    auto h_input = random_floats(n, -1.0f, 1.0f, 820);
    auto h_weight = random_floats(n, 0.8f, 1.2f, 821);
    float eps = 1e-5f;

    auto ref = cpu_rmsnorm(h_input, h_weight, eps);

    GPUMem d_in(n), d_w(n), d_out(n);
    d_in.upload(h_input.data(), n);
    d_w.upload(h_weight.data(), n);

    vulcan::cuda::launch_rmsnorm(d_in.ptr, d_w.ptr, d_out.ptr, n, eps);

    std::vector<float> h_out(n);
    d_out.download(h_out.data(), n);

    float err = max_abs_error(h_out, ref);
    EXPECT_LT(err, 1e-3f) << "Max RMSNorm error at dim=4096: " << err;
}

TEST(RMSNormTest, SmallDimension_32) {
    const int n = 32;
    auto h_input = random_floats(n, -3.0f, 3.0f, 830);
    auto h_weight = random_floats(n, 0.1f, 5.0f, 831);
    float eps = 1e-6f;

    auto ref = cpu_rmsnorm(h_input, h_weight, eps);

    GPUMem d_in(n), d_w(n), d_out(n);
    d_in.upload(h_input.data(), n);
    d_w.upload(h_weight.data(), n);

    vulcan::cuda::launch_rmsnorm(d_in.ptr, d_w.ptr, d_out.ptr, n, eps);

    std::vector<float> h_out(n);
    d_out.download(h_out.data(), n);

    float err = max_abs_error(h_out, ref);
    EXPECT_LT(err, 1e-5f) << "Max RMSNorm error at dim=32: " << err;
}

TEST(RMSNormTest, OutputRMSIsNormalized) {
    // After RMSNorm with unit weights, the RMS of output should be ~1.0
    const int n = 512;
    auto h_input = random_floats(n, -5.0f, 5.0f, 840);
    std::vector<float> h_weight(n, 1.0f);
    float eps = 1e-5f;

    GPUMem d_in(n), d_w(n), d_out(n);
    d_in.upload(h_input.data(), n);
    d_w.upload(h_weight.data(), n);

    vulcan::cuda::launch_rmsnorm(d_in.ptr, d_w.ptr, d_out.ptr, n, eps);

    std::vector<float> h_out(n);
    d_out.download(h_out.data(), n);

    // Compute RMS of output
    float ss = 0.0f;
    for (float x : h_out) ss += x * x;
    float output_rms = std::sqrt(ss / n);

    // RMS should be close to 1.0
    EXPECT_NEAR(output_rms, 1.0f, 0.01f)
        << "Output RMS should be ~1.0 after normalization with unit weights";
}

// ═══════════════════════════════════════════════════════════════════════════
//  In-Place Safety Test
// ═══════════════════════════════════════════════════════════════════════════

TEST(KernelTest, SiLUInPlaceSafe) {
    // Verify SiLU works when input != output (no aliasing issues)
    const int n = 1024;
    auto h_in = random_floats(n, -3.0f, 3.0f, 900);
    std::vector<float> ref(n);
    for (int i = 0; i < n; ++i) ref[i] = cpu_silu(h_in[i]);

    GPUMem d_in(n), d_out(n);
    d_in.upload(h_in.data(), n);

    vulcan::cuda::launch_silu(d_in.ptr, d_out.ptr, n);

    // Verify input buffer unchanged
    std::vector<float> h_in_check(n);
    d_in.download(h_in_check.data(), n);
    float in_err = max_abs_error(h_in_check, h_in);
    EXPECT_LT(in_err, 1e-7f) << "Input buffer was modified!";

    // Verify output
    std::vector<float> h_out(n);
    d_out.download(h_out.data(), n);
    float out_err = max_abs_error(h_out, ref);
    EXPECT_LT(out_err, 1e-5f);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Non-Aligned Size Edge Cases
// ═══════════════════════════════════════════════════════════════════════════

TEST(KernelTest, SiLU_NonAligned_999) {
    const int n = 999;  // Not a multiple of block size (256)
    auto h_in = random_floats(n, -5.0f, 5.0f, 950);
    std::vector<float> ref(n);
    for (int i = 0; i < n; ++i) ref[i] = cpu_silu(h_in[i]);

    GPUMem d_in(n), d_out(n);
    d_in.upload(h_in.data(), n);

    vulcan::cuda::launch_silu(d_in.ptr, d_out.ptr, n);

    std::vector<float> h_out(n);
    d_out.download(h_out.data(), n);

    float err = max_abs_error(h_out, ref);
    EXPECT_LT(err, 1e-5f) << "Non-aligned SiLU error: " << err;
}

TEST(KernelTest, ReLU_NonAligned_777) {
    const int n = 777;
    auto h_in = random_floats(n, -5.0f, 5.0f, 960);
    std::vector<float> ref(n);
    for (int i = 0; i < n; ++i) ref[i] = cpu_relu(h_in[i]);

    GPUMem d_in(n), d_out(n);
    d_in.upload(h_in.data(), n);

    vulcan::cuda::launch_relu(d_in.ptr, d_out.ptr, n);

    std::vector<float> h_out(n);
    d_out.download(h_out.data(), n);

    float err = max_abs_error(h_out, ref);
    EXPECT_LT(err, 1e-6f);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Softmax Tests
// ═══════════════════════════════════════════════════════════════════════════

/// CPU reference: numerically stable softmax
static std::vector<float> cpu_softmax(const std::vector<float>& input) {
    int n = static_cast<int>(input.size());
    float max_val = *std::max_element(input.begin(), input.end());

    std::vector<float> output(n);
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    for (int i = 0; i < n; ++i) output[i] /= sum;
    return output;
}

TEST(SoftmaxTest, KnownValues) {
    const int n = 4;
    std::vector<float> h_in = {1.0f, 2.0f, 3.0f, 4.0f};
    auto ref = cpu_softmax(h_in);

    GPUMem d_in(n), d_out(n);
    d_in.upload(h_in.data(), n);

    vulcan::cuda::launch_softmax(d_in.ptr, d_out.ptr, n);

    std::vector<float> h_out(n);
    d_out.download(h_out.data(), n);

    // Output should sum to 1
    float sum = 0.0f;
    for (float p : h_out) sum += p;
    EXPECT_NEAR(sum, 1.0f, 1e-5f) << "Softmax output must sum to 1";

    // Check individual values
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(h_out[i], ref[i], 1e-5f)
            << "Mismatch at index " << i;
    }
}

TEST(SoftmaxTest, LargeValues_NumericalStability) {
    // Test with large values that would overflow naive exp()
    const int n = 5;
    std::vector<float> h_in = {1000.0f, 1001.0f, 999.0f, 1000.5f, 998.0f};
    auto ref = cpu_softmax(h_in);

    GPUMem d_in(n), d_out(n);
    d_in.upload(h_in.data(), n);

    vulcan::cuda::launch_softmax(d_in.ptr, d_out.ptr, n);

    std::vector<float> h_out(n);
    d_out.download(h_out.data(), n);

    // No NaN or Inf
    for (int i = 0; i < n; ++i) {
        EXPECT_FALSE(std::isnan(h_out[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(h_out[i])) << "Inf at index " << i;
    }

    float err = max_abs_error(h_out, ref);
    EXPECT_LT(err, 1e-5f) << "Softmax with large values, max error: " << err;
}

TEST(SoftmaxTest, UniformDistribution) {
    // All equal inputs → uniform output
    const int n = 128;
    std::vector<float> h_in(n, 5.0f);

    GPUMem d_in(n), d_out(n);
    d_in.upload(h_in.data(), n);

    vulcan::cuda::launch_softmax(d_in.ptr, d_out.ptr, n);

    std::vector<float> h_out(n);
    d_out.download(h_out.data(), n);

    float expected = 1.0f / n;
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(h_out[i], expected, 1e-5f);
    }
}

TEST(SoftmaxTest, VocabSize_32000) {
    // Llama-2 vocab size
    const int n = 32000;
    auto h_in = random_floats(n, -5.0f, 5.0f, 1000);
    auto ref = cpu_softmax(h_in);

    GPUMem d_in(n), d_out(n);
    d_in.upload(h_in.data(), n);

    vulcan::cuda::launch_softmax(d_in.ptr, d_out.ptr, n);

    std::vector<float> h_out(n);
    d_out.download(h_out.data(), n);

    // Sum should be 1
    float sum = 0.0f;
    for (float p : h_out) sum += p;
    EXPECT_NEAR(sum, 1.0f, 1e-3f);

    float err = max_abs_error(h_out, ref);
    EXPECT_LT(err, 1e-4f) << "Softmax at vocab size, max error: " << err;
}

// ═══════════════════════════════════════════════════════════════════════════
//  RoPE (Rotary Position Embedding) Tests
// ═══════════════════════════════════════════════════════════════════════════

/// CPU reference: RoPE
static std::vector<float> cpu_rope(const std::vector<float>& input,
                                   int pos, int head_dim,
                                   float theta_base = 10000.0f) {
    std::vector<float> output(head_dim);
    int half_dim = head_dim / 2;

    for (int i = 0; i < half_dim; ++i) {
        float freq = 1.0f / std::pow(theta_base, 2.0f * i / static_cast<float>(head_dim));
        float theta = pos * freq;
        float cos_t = std::cos(theta);
        float sin_t = std::sin(theta);

        float x0 = input[2 * i];
        float x1 = input[2 * i + 1];

        output[2 * i]     = x0 * cos_t - x1 * sin_t;
        output[2 * i + 1] = x0 * sin_t + x1 * cos_t;
    }
    return output;
}

TEST(RoPETest, Position0_Identity) {
    // At position 0, all thetas are 0 → cos=1, sin=0 → output = input
    const int head_dim = 64;
    auto h_in = random_floats(head_dim, -1.0f, 1.0f, 1100);

    GPUMem d_in(head_dim), d_out(head_dim);
    d_in.upload(h_in.data(), head_dim);

    vulcan::cuda::launch_rope(d_in.ptr, d_out.ptr, 0, head_dim, 10000.0f);

    std::vector<float> h_out(head_dim);
    d_out.download(h_out.data(), head_dim);

    float err = max_abs_error(h_out, h_in);
    EXPECT_LT(err, 1e-5f) << "RoPE at pos=0 should be identity. Max error: " << err;
}

TEST(RoPETest, PreservesNorm) {
    // RoPE is a rotation → should preserve the L2 norm of each pair
    const int head_dim = 128;
    auto h_in = random_floats(head_dim, -2.0f, 2.0f, 1200);

    GPUMem d_in(head_dim), d_out(head_dim);
    d_in.upload(h_in.data(), head_dim);

    vulcan::cuda::launch_rope(d_in.ptr, d_out.ptr, 42, head_dim, 10000.0f);

    std::vector<float> h_out(head_dim);
    d_out.download(h_out.data(), head_dim);

    // Check each pair preserves norm
    for (int i = 0; i < head_dim / 2; ++i) {
        float in_norm = std::sqrt(h_in[2*i]*h_in[2*i] + h_in[2*i+1]*h_in[2*i+1]);
        float out_norm = std::sqrt(h_out[2*i]*h_out[2*i] + h_out[2*i+1]*h_out[2*i+1]);
        EXPECT_NEAR(in_norm, out_norm, 1e-4f)
            << "Pair " << i << ": norm not preserved";
    }
}

TEST(RoPETest, AgainstCPUReference) {
    const int head_dim = 128;
    const int pos = 100;  // Test at a non-trivial position
    auto h_in = random_floats(head_dim, -1.0f, 1.0f, 1300);
    auto ref = cpu_rope(h_in, pos, head_dim);

    GPUMem d_in(head_dim), d_out(head_dim);
    d_in.upload(h_in.data(), head_dim);

    vulcan::cuda::launch_rope(d_in.ptr, d_out.ptr, pos, head_dim, 10000.0f);

    std::vector<float> h_out(head_dim);
    d_out.download(h_out.data(), head_dim);

    float err = max_abs_error(h_out, ref);
    EXPECT_LT(err, 1e-4f) << "RoPE vs CPU reference, max error: " << err;
}

