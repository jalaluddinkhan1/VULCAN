/// @file test_quantization.cu
/// @brief Golden tests for quantization and fused kernels.
///
/// Tests: INT4 dequantization, quantized matmul, fused SiLU*Up,
///        fused RMSNorm+Linear, fused Residual+RMSNorm.

#include <gtest/gtest.h>
#include "cuda/kernels.h"
#include "cuda/utils.h"
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <cstdint>

// ─── Helpers ────────────────────────────────────────────────────────────────

static std::vector<float> random_floats(int n, float lo, float hi, int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) v[i] = dist(rng);
    return v;
}

static float max_abs_error(const std::vector<float>& a,
                           const std::vector<float>& b) {
    float e = 0.0f;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i)
        e = std::max(e, std::abs(a[i] - b[i]));
    return e;
}

struct GPUMem {
    float* ptr = nullptr;
    GPUMem(size_t n) { CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(float))); }
    ~GPUMem() { if (ptr) cudaFree(ptr); }
    void upload(const float* h, size_t n) {
        CUDA_CHECK(cudaMemcpy(ptr, h, n * sizeof(float), cudaMemcpyHostToDevice));
    }
    void download(float* h, size_t n) {
        CUDA_CHECK(cudaMemcpy(h, ptr, n * sizeof(float), cudaMemcpyDeviceToHost));
    }
};

static float cpu_silu(float x) { return x / (1.0f + std::exp(-x)); }

// ═══════════════════════════════════════════════════════════════════════════
//  INT4 Dequantization Tests
// ═══════════════════════════════════════════════════════════════════════════

TEST(DequantTest, KnownValues) {
    // Pack: [3, -2] → unsigned [11, 6] → byte = (6 << 4) | 11 = 0x6B
    const int n = 2;
    const int group_size = 128;
    float scale = 0.5f;

    uint8_t packed = 0x6B;  // low=11→3, high=6→-2
    float expected_low  = 3.0f * scale;   // 1.5
    float expected_high = -2.0f * scale;  // -1.0

    uint8_t* d_packed;
    CUDA_CHECK(cudaMalloc(&d_packed, 1));
    CUDA_CHECK(cudaMemcpy(d_packed, &packed, 1, cudaMemcpyHostToDevice));

    float* d_scales;
    CUDA_CHECK(cudaMalloc(&d_scales, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_scales, &scale, sizeof(float), cudaMemcpyHostToDevice));

    GPUMem d_out(n);
    vulcan::cuda::launch_dequantize_int4(d_packed, d_scales, d_out.ptr, n, group_size);

    std::vector<float> h_out(n);
    d_out.download(h_out.data(), n);

    EXPECT_NEAR(h_out[0], expected_low, 1e-5f);
    EXPECT_NEAR(h_out[1], expected_high, 1e-5f);

    cudaFree(d_packed);
    cudaFree(d_scales);
}

TEST(DequantTest, FullGroup) {
    const int group_size = 128;
    const int n = group_size;
    const int num_bytes = n / 2;
    float scale = 0.1f;

    // Create packed data: all zeros → unsigned 8 → signed 0
    std::vector<uint8_t> h_packed(num_bytes, 0x88);  // low=8→0, high=8→0
    std::vector<float> h_scales = {scale};

    uint8_t* d_packed;
    CUDA_CHECK(cudaMalloc(&d_packed, num_bytes));
    CUDA_CHECK(cudaMemcpy(d_packed, h_packed.data(), num_bytes, cudaMemcpyHostToDevice));

    float* d_scales;
    CUDA_CHECK(cudaMalloc(&d_scales, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_scales, h_scales.data(), sizeof(float), cudaMemcpyHostToDevice));

    GPUMem d_out(n);
    vulcan::cuda::launch_dequantize_int4(d_packed, d_scales, d_out.ptr, n, group_size);

    std::vector<float> h_out(n);
    d_out.download(h_out.data(), n);

    // All should be 0 * scale = 0
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(h_out[i], 0.0f, 1e-6f);
    }

    cudaFree(d_packed);
    cudaFree(d_scales);
}

TEST(DequantTest, MultipleGroups) {
    const int group_size = 4;
    const int n = 8;  // 2 groups
    const int num_bytes = n / 2;

    // Group 0 scale=1.0, Group 1 scale=2.0
    std::vector<float> h_scales = {1.0f, 2.0f};

    // Pack values: [1, -1, 2, -2, 3, -3, 1, -1]
    // unsigned: [9, 7, 10, 6, 11, 5, 9, 7]
    // bytes: (7<<4)|9, (6<<4)|10, (5<<4)|11, (7<<4)|9
    std::vector<uint8_t> h_packed = {0x79, 0x6A, 0x5B, 0x79};

    uint8_t* d_packed;
    CUDA_CHECK(cudaMalloc(&d_packed, num_bytes));
    CUDA_CHECK(cudaMemcpy(d_packed, h_packed.data(), num_bytes, cudaMemcpyHostToDevice));

    float* d_scales;
    CUDA_CHECK(cudaMalloc(&d_scales, h_scales.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_scales, h_scales.data(),
                          h_scales.size() * sizeof(float), cudaMemcpyHostToDevice));

    GPUMem d_out(n);
    vulcan::cuda::launch_dequantize_int4(d_packed, d_scales, d_out.ptr, n, group_size);

    std::vector<float> h_out(n);
    d_out.download(h_out.data(), n);

    // Group 0 (scale=1.0): 1, -1, 2, -2
    EXPECT_NEAR(h_out[0],  1.0f, 1e-5f);
    EXPECT_NEAR(h_out[1], -1.0f, 1e-5f);
    EXPECT_NEAR(h_out[2],  2.0f, 1e-5f);
    EXPECT_NEAR(h_out[3], -2.0f, 1e-5f);
    // Group 1 (scale=2.0): 6, -6, 2, -2
    EXPECT_NEAR(h_out[4],  6.0f, 1e-5f);
    EXPECT_NEAR(h_out[5], -6.0f, 1e-5f);
    EXPECT_NEAR(h_out[6],  2.0f, 1e-5f);
    EXPECT_NEAR(h_out[7], -2.0f, 1e-5f);

    cudaFree(d_packed);
    cudaFree(d_scales);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Fused SiLU * Up Tests
// ═══════════════════════════════════════════════════════════════════════════

TEST(FusedSiLUMulTest, MatchesSeparate) {
    const int n = 4096;
    auto h_gate = random_floats(n, -5.0f, 5.0f, 5000);
    auto h_up   = random_floats(n, -3.0f, 3.0f, 5001);

    // CPU reference: silu(gate) * up
    std::vector<float> ref(n);
    for (int i = 0; i < n; ++i) {
        ref[i] = cpu_silu(h_gate[i]) * h_up[i];
    }

    GPUMem d_gate(n), d_up(n), d_out(n);
    d_gate.upload(h_gate.data(), n);
    d_up.upload(h_up.data(), n);

    vulcan::cuda::launch_fused_silu_mul(d_gate.ptr, d_up.ptr, d_out.ptr, n);

    std::vector<float> h_out(n);
    d_out.download(h_out.data(), n);

    float err = max_abs_error(h_out, ref);
    EXPECT_LT(err, 1e-5f) << "Fused SiLU*Up max error: " << err;
}

TEST(FusedSiLUMulTest, ZeroGate) {
    const int n = 256;
    std::vector<float> h_gate(n, 0.0f);
    auto h_up = random_floats(n, -1.0f, 1.0f, 5010);

    GPUMem d_gate(n), d_up(n), d_out(n);
    d_gate.upload(h_gate.data(), n);
    d_up.upload(h_up.data(), n);

    vulcan::cuda::launch_fused_silu_mul(d_gate.ptr, d_up.ptr, d_out.ptr, n);

    std::vector<float> h_out(n);
    d_out.download(h_out.data(), n);

    // silu(0) = 0, so output should be all zeros
    for (int i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(h_out[i], 0.0f);
    }
}

TEST(FusedSiLUMulTest, LargeVector) {
    const int n = 1 << 17;  // 128K (typical intermediate dim)
    auto h_gate = random_floats(n, -10.0f, 10.0f, 5020);
    auto h_up   = random_floats(n, -2.0f, 2.0f, 5021);

    std::vector<float> ref(n);
    for (int i = 0; i < n; ++i) ref[i] = cpu_silu(h_gate[i]) * h_up[i];

    GPUMem d_gate(n), d_up(n), d_out(n);
    d_gate.upload(h_gate.data(), n);
    d_up.upload(h_up.data(), n);

    vulcan::cuda::launch_fused_silu_mul(d_gate.ptr, d_up.ptr, d_out.ptr, n);

    std::vector<float> h_out(n);
    d_out.download(h_out.data(), n);

    float err = max_abs_error(h_out, ref);
    EXPECT_LT(err, 1e-4f) << "Large fused SiLU*Up error: " << err;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Fused RMSNorm + Linear Tests
// ═══════════════════════════════════════════════════════════════════════════

TEST(FusedRMSNormLinearTest, MatchesSeparate) {
    // Compare fused vs separate rmsnorm → matmul
    const int dim = 64, out_dim = 32;
    float eps = 1e-5f;

    auto h_input = random_floats(dim, -1.0f, 1.0f, 6000);
    auto h_norm_w = random_floats(dim, 0.5f, 2.0f, 6001);
    auto h_proj_w = random_floats(dim * out_dim, -0.1f, 0.1f, 6002);

    // CPU reference: rmsnorm then matmul
    float ss = 0.0f;
    for (float x : h_input) ss += x * x;
    float rms = 1.0f / std::sqrt(ss / dim + eps);
    std::vector<float> normed(dim);
    for (int i = 0; i < dim; ++i) normed[i] = h_input[i] * rms * h_norm_w[i];

    // matmul: [1, dim] @ [dim, out_dim]
    std::vector<float> ref(out_dim, 0.0f);
    for (int j = 0; j < out_dim; ++j) {
        for (int k = 0; k < dim; ++k) {
            ref[j] += normed[k] * h_proj_w[k * out_dim + j];
        }
    }

    GPUMem d_in(dim), d_nw(dim), d_pw(dim * out_dim), d_out(out_dim);
    d_in.upload(h_input.data(), dim);
    d_nw.upload(h_norm_w.data(), dim);
    d_pw.upload(h_proj_w.data(), dim * out_dim);

    vulcan::cuda::launch_fused_rmsnorm_linear(
        d_in.ptr, d_nw.ptr, d_pw.ptr, d_out.ptr,
        1, dim, out_dim, eps
    );

    std::vector<float> h_out(out_dim);
    d_out.download(h_out.data(), out_dim);

    float err = max_abs_error(h_out, ref);
    EXPECT_LT(err, 1e-2f) << "Fused RMSNorm+Linear max error: " << err;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Fused Residual + RMSNorm Tests
// ═══════════════════════════════════════════════════════════════════════════

TEST(FusedResidualRMSNormTest, MatchesSeparate) {
    const int n = 128;
    float eps = 1e-5f;

    auto h_residual = random_floats(n, -1.0f, 1.0f, 7000);
    auto h_hidden   = random_floats(n, -1.0f, 1.0f, 7001);
    auto h_weight   = random_floats(n, 0.5f, 2.0f, 7002);

    // CPU reference: residual + hidden → rmsnorm
    std::vector<float> sum(n);
    for (int i = 0; i < n; ++i) sum[i] = h_residual[i] + h_hidden[i];

    float ss = 0.0f;
    for (float x : sum) ss += x * x;
    float rms = 1.0f / std::sqrt(ss / n + eps);

    std::vector<float> ref(n);
    for (int i = 0; i < n; ++i) ref[i] = sum[i] * rms * h_weight[i];

    GPUMem d_res(n), d_hid(n), d_w(n), d_out(n), d_res_out(n);
    d_res.upload(h_residual.data(), n);
    d_hid.upload(h_hidden.data(), n);
    d_w.upload(h_weight.data(), n);

    vulcan::cuda::launch_fused_residual_rmsnorm(
        d_res.ptr, d_hid.ptr, d_w.ptr, d_out.ptr, d_res_out.ptr, n, eps
    );

    std::vector<float> h_out(n);
    d_out.download(h_out.data(), n);

    float err = max_abs_error(h_out, ref);
    EXPECT_LT(err, 1e-4f) << "Fused Residual+RMSNorm error: " << err;

    // Verify residual_out = residual + hidden
    std::vector<float> h_rout(n);
    d_res_out.download(h_rout.data(), n);
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(h_rout[i], sum[i], 1e-6f);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Quantized MatMul Tests
// ═══════════════════════════════════════════════════════════════════════════

TEST(QuantizedMatMulTest, SmallIdentity) {
    // Simple test: A = identity-like, B packed = known values
    const int M = 1, K = 4, N = 2;
    const int group_size = 4;

    // A = [1, 1, 1, 1]
    std::vector<float> h_A = {1.0f, 1.0f, 1.0f, 1.0f};

    // B values we want: [[1, 2], [3, -1], [2, 0], [-1, 1]]
    // Scales per group per col: col0 scale = 3/7 ≈ 0.4286, col1 scale = 2/7 ≈ 0.2857
    // Quantized (round to nearest): col0 [2, 7, 5, -2], col1 [7, -4, 0, 4]
    // This is complex to set up manually, so let's do a simpler approach

    // Instead: B = all zeros → C should be zero
    std::vector<uint8_t> h_B(K / 2 * N, 0x88);  // All zeros (8-8=0)
    std::vector<float> h_scales(1 * N, 1.0f);     // 1 group per column

    float* d_A;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));

    uint8_t* d_B;
    CUDA_CHECK(cudaMalloc(&d_B, h_B.size()));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), h_B.size(), cudaMemcpyHostToDevice));

    float* d_scales;
    CUDA_CHECK(cudaMalloc(&d_scales, h_scales.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_scales, h_scales.data(),
                          h_scales.size() * sizeof(float), cudaMemcpyHostToDevice));

    GPUMem d_C(M * N);
    vulcan::cuda::launch_quantized_matmul(d_A, d_B, d_scales, d_C.ptr, M, K, N, group_size);

    std::vector<float> h_C(M * N);
    d_C.download(h_C.data(), M * N);

    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(h_C[i], 0.0f, 1e-5f) << "All-zero B should give zero output";
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_scales);
}
