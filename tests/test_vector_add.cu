/// @file test_vector_add.cu
/// @brief Unit test for the vector_add CUDA kernel.
///
/// Verification test: ensures the CUDA build pipeline works
/// and kernels produce correct results.

#include <gtest/gtest.h>
#include "cuda/kernels.h"
#include "cuda/utils.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <random>

using namespace vulcan::cuda;

// ─── Helper ─────────────────────────────────────────────────────────────────

/// Generate a vector of random floats in [lo, hi].
static std::vector<float> random_floats(int n, float lo = -10.0f,
                                        float hi = 10.0f) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) v[i] = dist(rng);
    return v;
}

// ─── Tests ──────────────────────────────────────────────────────────────────

TEST(VectorAddTest, SmallVector) {
    const int n = 4;
    std::vector<float> h_a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> h_b = {10.0f, 20.0f, 30.0f, 40.0f};
    std::vector<float> h_c(n);

    // Allocate GPU memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, n * sizeof(float)));

    // Copy inputs to GPU
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    launch_vector_add(d_a, d_b, d_c, n);

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify
    EXPECT_FLOAT_EQ(h_c[0], 11.0f);
    EXPECT_FLOAT_EQ(h_c[1], 22.0f);
    EXPECT_FLOAT_EQ(h_c[2], 33.0f);
    EXPECT_FLOAT_EQ(h_c[3], 44.0f);

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

TEST(VectorAddTest, LargeRandomVector) {
    const int n = 1 << 20;  // 1M elements
    auto h_a = random_floats(n);
    auto h_b = random_floats(n, -5.0f, 5.0f);
    std::vector<float> h_c(n);

    // Compute CPU reference
    std::vector<float> ref(n);
    for (int i = 0; i < n; ++i) ref[i] = h_a[i] + h_b[i];

    // GPU allocation
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    // Launch
    launch_vector_add(d_a, d_b, d_c, n);

    // Read back
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify: max absolute error < 1e-5
    float max_error = 0.0f;
    for (int i = 0; i < n; ++i) {
        float err = std::abs(h_c[i] - ref[i]);
        if (err > max_error) max_error = err;
    }
    EXPECT_LT(max_error, 1e-5f) << "Max error: " << max_error;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

TEST(VectorAddTest, NonAlignedSize) {
    // Test with a size that's not a multiple of block size (256)
    const int n = 1000;
    auto h_a = random_floats(n);
    auto h_b = random_floats(n);
    std::vector<float> h_c(n);

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    launch_vector_add(d_a, d_b, d_c, n);

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(h_c[i], h_a[i] + h_b[i], 1e-5f);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

TEST(VectorAddTest, SingleElement) {
    const int n = 1;
    float h_a = 3.14f, h_b = 2.71f, h_c = 0.0f;

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_a, &h_a, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &h_b, sizeof(float), cudaMemcpyHostToDevice));

    launch_vector_add(d_a, d_b, d_c, n);

    CUDA_CHECK(cudaMemcpy(&h_c, d_c, sizeof(float), cudaMemcpyDeviceToHost));

    EXPECT_NEAR(h_c, 5.85f, 1e-5f);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
