/// @file test_attention.cu
/// @brief Golden tests for Scaled Dot-Product Attention.
///
/// Tests the attention kernel against CPU reference implementation.
/// Validates causal masking, numerical stability, and correctness.

#include <gtest/gtest.h>
#include "cuda/kernels.h"
#include "cuda/utils.h"
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <cmath>
#include <cfloat>
#include <algorithm>

// ─── Helpers ────────────────────────────────────────────────────────────────

static std::vector<float> random_floats(int n, float lo = -1.0f,
                                        float hi = 1.0f, int seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) v[i] = dist(rng);
    return v;
}

static float max_abs_error(const std::vector<float>& a,
                           const std::vector<float>& b) {
    float max_err = 0.0f;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        max_err = std::max(max_err, std::abs(a[i] - b[i]));
    }
    return max_err;
}

/// RAII GPU memory wrapper
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

/// CPU reference: Causal Scaled Dot-Product Attention
/// Layout: [batch, heads, seq_len, head_dim], batch and heads fixed at
/// the outer loops.
static std::vector<float> cpu_attention(const std::vector<float>& Q,
                                        const std::vector<float>& K,
                                        const std::vector<float>& V,
                                        int batch, int heads,
                                        int seq_len, int head_dim) {
    int total = batch * heads * seq_len * head_dim;
    std::vector<float> output(total, 0.0f);
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            int bh_off = ((b * heads + h) * seq_len) * head_dim;

            for (int row = 0; row < seq_len; ++row) {
                // Compute scores with causal mask
                std::vector<float> scores(seq_len, -FLT_MAX);
                for (int j = 0; j <= row; ++j) {  // Causal: j <= row
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        dot += Q[bh_off + row * head_dim + d]
                             * K[bh_off + j * head_dim + d];
                    }
                    scores[j] = dot * scale;
                }

                // Softmax (numerically stable)
                float max_s = *std::max_element(scores.begin(),
                    scores.begin() + row + 1);
                float sum = 0.0f;
                for (int j = 0; j <= row; ++j) {
                    scores[j] = std::exp(scores[j] - max_s);
                    sum += scores[j];
                }
                for (int j = 0; j <= row; ++j) {
                    scores[j] /= sum;
                }

                // Weighted sum of V
                for (int d = 0; d < head_dim; ++d) {
                    float val = 0.0f;
                    for (int j = 0; j <= row; ++j) {
                        val += scores[j] * V[bh_off + j * head_dim + d];
                    }
                    output[bh_off + row * head_dim + d] = val;
                }
            }
        }
    }

    return output;
}

// ─── Tests ──────────────────────────────────────────────────────────────────

TEST(AttentionTest, SingleHead_SmallSeq) {
    // batch=1, heads=1, seq_len=4, head_dim=8
    const int B = 1, H = 1, S = 4, D = 8;
    int total = B * H * S * D;

    auto h_Q = random_floats(total, -1.0f, 1.0f, 2000);
    auto h_K = random_floats(total, -1.0f, 1.0f, 2001);
    auto h_V = random_floats(total, -1.0f, 1.0f, 2002);
    auto ref = cpu_attention(h_Q, h_K, h_V, B, H, S, D);

    GPUMem d_Q(total), d_K(total), d_V(total), d_O(total);
    d_Q.upload(h_Q.data(), total);
    d_K.upload(h_K.data(), total);
    d_V.upload(h_V.data(), total);

    vulcan::cuda::launch_attention(d_Q.ptr, d_K.ptr, d_V.ptr, d_O.ptr,
                                   B, H, S, D);

    std::vector<float> h_O(total);
    d_O.download(h_O.data(), total);

    float err = max_abs_error(h_O, ref);
    EXPECT_LT(err, 1e-4f) << "Single head attention error: " << err;
}

TEST(AttentionTest, MultiHead_4Heads) {
    const int B = 1, H = 4, S = 8, D = 16;
    int total = B * H * S * D;

    auto h_Q = random_floats(total, -0.5f, 0.5f, 2010);
    auto h_K = random_floats(total, -0.5f, 0.5f, 2011);
    auto h_V = random_floats(total, -0.5f, 0.5f, 2012);
    auto ref = cpu_attention(h_Q, h_K, h_V, B, H, S, D);

    GPUMem d_Q(total), d_K(total), d_V(total), d_O(total);
    d_Q.upload(h_Q.data(), total);
    d_K.upload(h_K.data(), total);
    d_V.upload(h_V.data(), total);

    vulcan::cuda::launch_attention(d_Q.ptr, d_K.ptr, d_V.ptr, d_O.ptr,
                                   B, H, S, D);

    std::vector<float> h_O(total);
    d_O.download(h_O.data(), total);

    float err = max_abs_error(h_O, ref);
    EXPECT_LT(err, 1e-3f) << "Multi-head attention error: " << err;
}

TEST(AttentionTest, CausalMask_FirstRow) {
    // The first row (position 0) should only attend to itself
    // This means the output should equal V[0] exactly
    const int B = 1, H = 1, S = 4, D = 4;
    int total = B * H * S * D;

    auto h_Q = random_floats(total, -1.0f, 1.0f, 2020);
    auto h_K = random_floats(total, -1.0f, 1.0f, 2021);
    auto h_V = random_floats(total, -1.0f, 1.0f, 2022);

    GPUMem d_Q(total), d_K(total), d_V(total), d_O(total);
    d_Q.upload(h_Q.data(), total);
    d_K.upload(h_K.data(), total);
    d_V.upload(h_V.data(), total);

    vulcan::cuda::launch_attention(d_Q.ptr, d_K.ptr, d_V.ptr, d_O.ptr,
                                   B, H, S, D);

    std::vector<float> h_O(total);
    d_O.download(h_O.data(), total);

    // Position 0: softmax over single element = 1.0, so output = V[0]
    for (int d = 0; d < D; ++d) {
        EXPECT_NEAR(h_O[d], h_V[d], 1e-5f)
            << "First row should equal V[0] exactly (causal mask)";
    }
}

TEST(AttentionTest, LongerSequence_32) {
    const int B = 1, H = 2, S = 32, D = 32;
    int total = B * H * S * D;

    auto h_Q = random_floats(total, -0.3f, 0.3f, 2030);
    auto h_K = random_floats(total, -0.3f, 0.3f, 2031);
    auto h_V = random_floats(total, -0.3f, 0.3f, 2032);
    auto ref = cpu_attention(h_Q, h_K, h_V, B, H, S, D);

    GPUMem d_Q(total), d_K(total), d_V(total), d_O(total);
    d_Q.upload(h_Q.data(), total);
    d_K.upload(h_K.data(), total);
    d_V.upload(h_V.data(), total);

    vulcan::cuda::launch_attention(d_Q.ptr, d_K.ptr, d_V.ptr, d_O.ptr,
                                   B, H, S, D);

    std::vector<float> h_O(total);
    d_O.download(h_O.data(), total);

    float err = max_abs_error(h_O, ref);
    EXPECT_LT(err, 1e-3f) << "Seq=32 attention error: " << err;
}

TEST(AttentionTest, Batched_2x4Heads) {
    const int B = 2, H = 4, S = 8, D = 16;
    int total = B * H * S * D;

    auto h_Q = random_floats(total, -0.5f, 0.5f, 2040);
    auto h_K = random_floats(total, -0.5f, 0.5f, 2041);
    auto h_V = random_floats(total, -0.5f, 0.5f, 2042);
    auto ref = cpu_attention(h_Q, h_K, h_V, B, H, S, D);

    GPUMem d_Q(total), d_K(total), d_V(total), d_O(total);
    d_Q.upload(h_Q.data(), total);
    d_K.upload(h_K.data(), total);
    d_V.upload(h_V.data(), total);

    vulcan::cuda::launch_attention(d_Q.ptr, d_K.ptr, d_V.ptr, d_O.ptr,
                                   B, H, S, D);

    std::vector<float> h_O(total);
    d_O.download(h_O.data(), total);

    float err = max_abs_error(h_O, ref);
    EXPECT_LT(err, 1e-3f) << "Batched attention error: " << err;
}

TEST(AttentionTest, OutputSumMatchesV) {
    // Each output row should be a convex combination of V rows
    // (softmax weights sum to 1, all positive)
    // So output values should be bounded by min/max of V values
    const int B = 1, H = 1, S = 8, D = 8;
    int total = B * H * S * D;

    auto h_Q = random_floats(total, -1.0f, 1.0f, 2050);
    auto h_K = random_floats(total, -1.0f, 1.0f, 2051);
    auto h_V = random_floats(total, -1.0f, 1.0f, 2052);

    GPUMem d_Q(total), d_K(total), d_V(total), d_O(total);
    d_Q.upload(h_Q.data(), total);
    d_K.upload(h_K.data(), total);
    d_V.upload(h_V.data(), total);

    vulcan::cuda::launch_attention(d_Q.ptr, d_K.ptr, d_V.ptr, d_O.ptr,
                                   B, H, S, D);

    std::vector<float> h_O(total);
    d_O.download(h_O.data(), total);

    float v_min = *std::min_element(h_V.begin(), h_V.end());
    float v_max = *std::max_element(h_V.begin(), h_V.end());

    // Every output value should be within [v_min, v_max]
    for (int i = 0; i < total; ++i) {
        EXPECT_GE(h_O[i], v_min - 1e-4f)
            << "Output below V min at index " << i;
        EXPECT_LE(h_O[i], v_max + 1e-4f)
            << "Output above V max at index " << i;
    }
}

// ─── Element-wise Kernel Tests ──────────────────────────────────────────────

TEST(ElementwiseTest, ResidualAdd) {
    const int n = 4096;
    auto h_a = random_floats(n, -2.0f, 2.0f, 3000);
    auto h_b = random_floats(n, -2.0f, 2.0f, 3001);

    GPUMem d_a(n), d_b(n), d_c(n);
    d_a.upload(h_a.data(), n);
    d_b.upload(h_b.data(), n);

    vulcan::cuda::launch_residual_add(d_a.ptr, d_b.ptr, d_c.ptr, n);

    std::vector<float> h_c(n);
    d_c.download(h_c.data(), n);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(h_c[i], h_a[i] + h_b[i], 1e-6f);
    }
}

TEST(ElementwiseTest, ElementMul) {
    const int n = 2048;
    auto h_a = random_floats(n, -3.0f, 3.0f, 3010);
    auto h_b = random_floats(n, -3.0f, 3.0f, 3011);

    GPUMem d_a(n), d_b(n), d_c(n);
    d_a.upload(h_a.data(), n);
    d_b.upload(h_b.data(), n);

    vulcan::cuda::launch_elementwise_mul(d_a.ptr, d_b.ptr, d_c.ptr, n);

    std::vector<float> h_c(n);
    d_c.download(h_c.data(), n);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(h_c[i], h_a[i] * h_b[i], 1e-5f);
    }
}

TEST(ElementwiseTest, EmbeddingLookup) {
    const int vocab = 100, dim = 16, seq_len = 4;

    // Create embedding table
    auto h_table = random_floats(vocab * dim, -1.0f, 1.0f, 3020);
    std::vector<int> h_ids = {5, 10, 3, 42};

    // Upload
    float* d_table;
    CUDA_CHECK(cudaMalloc(&d_table, vocab * dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_table, h_table.data(),
                          vocab * dim * sizeof(float), cudaMemcpyHostToDevice));

    int* d_ids;
    CUDA_CHECK(cudaMalloc(&d_ids, seq_len * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_ids, h_ids.data(),
                          seq_len * sizeof(int), cudaMemcpyHostToDevice));

    GPUMem d_out(seq_len * dim);

    vulcan::cuda::launch_embedding_lookup(d_table, d_ids,
                                         d_out.ptr, seq_len, dim);

    std::vector<float> h_out(seq_len * dim);
    d_out.download(h_out.data(), seq_len * dim);

    // Verify each token maps to the correct embedding row
    for (int t = 0; t < seq_len; ++t) {
        for (int d = 0; d < dim; ++d) {
            float expected = h_table[h_ids[t] * dim + d];
            EXPECT_FLOAT_EQ(h_out[t * dim + d], expected)
                << "Token " << t << " (id=" << h_ids[t] << ") dim " << d;
        }
    }

    cudaFree(d_table);
    cudaFree(d_ids);
}
