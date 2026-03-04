/// @file test_tensor.cpp
/// @brief Unit tests for vulcan::Tensor class.

#include <gtest/gtest.h>
#include "vulcan/tensor.h"
#include <vector>
#include <cstring>

using namespace vulcan;

// ─── Construction ───────────────────────────────────────────────────────────

TEST(TensorTest, DefaultConstruction) {
    Tensor t;
    EXPECT_FALSE(t.is_valid());
    EXPECT_EQ(t.numel(), 0u);
}

TEST(TensorTest, CPUConstruction) {
    Tensor t({2, 3, 4}, Device::CPU);
    EXPECT_TRUE(t.is_valid());
    EXPECT_EQ(t.numel(), 24u);
    EXPECT_EQ(t.ndim(), 3);
    EXPECT_EQ(t.size(0), 2);
    EXPECT_EQ(t.size(1), 3);
    EXPECT_EQ(t.size(2), 4);
    EXPECT_EQ(t.nbytes(), 24 * sizeof(float));
    EXPECT_EQ(t.device(), Device::CPU);
    EXPECT_NE(t.data(), nullptr);
}

TEST(TensorTest, CUDAConstruction) {
    Tensor t({256}, Device::CUDA);
    EXPECT_TRUE(t.is_valid());
    EXPECT_EQ(t.numel(), 256u);
    EXPECT_EQ(t.device(), Device::CUDA);
    EXPECT_NE(t.data(), nullptr);
}

// ─── Data Loading ───────────────────────────────────────────────────────────

TEST(TensorTest, FromHostCPU) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor t({4}, Device::CPU);
    t.from_host(data.data());

    EXPECT_FLOAT_EQ(t.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(t.data()[1], 2.0f);
    EXPECT_FLOAT_EQ(t.data()[2], 3.0f);
    EXPECT_FLOAT_EQ(t.data()[3], 4.0f);
}

TEST(TensorTest, FromHostGPU) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor t({4}, Device::CUDA);
    t.from_host(data.data());

    // Read back to verify
    std::vector<float> result(4);
    t.to_host(result.data());

    EXPECT_FLOAT_EQ(result[0], 1.0f);
    EXPECT_FLOAT_EQ(result[1], 2.0f);
    EXPECT_FLOAT_EQ(result[2], 3.0f);
    EXPECT_FLOAT_EQ(result[3], 4.0f);
}

// ─── Device Transfer ────────────────────────────────────────────────────────

TEST(TensorTest, CPUToGPUTransfer) {
    std::vector<float> data = {1.5f, 2.5f, 3.5f};
    Tensor cpu_tensor({3}, Device::CPU);
    cpu_tensor.from_host(data.data());

    Tensor gpu_tensor = cpu_tensor.to(Device::CUDA);
    EXPECT_EQ(gpu_tensor.device(), Device::CUDA);
    EXPECT_EQ(gpu_tensor.numel(), 3u);

    // Read back from GPU to verify
    std::vector<float> result(3);
    gpu_tensor.to_host(result.data());

    EXPECT_FLOAT_EQ(result[0], 1.5f);
    EXPECT_FLOAT_EQ(result[1], 2.5f);
    EXPECT_FLOAT_EQ(result[2], 3.5f);
}

TEST(TensorTest, GPUToCPUTransfer) {
    std::vector<float> data = {10.0f, 20.0f};
    Tensor gpu_tensor({2}, Device::CUDA);
    gpu_tensor.from_host(data.data());

    Tensor cpu_tensor = gpu_tensor.to(Device::CPU);
    EXPECT_EQ(cpu_tensor.device(), Device::CPU);

    EXPECT_FLOAT_EQ(cpu_tensor.data()[0], 10.0f);
    EXPECT_FLOAT_EQ(cpu_tensor.data()[1], 20.0f);
}

// ─── Move Semantics ─────────────────────────────────────────────────────────

TEST(TensorTest, MoveConstruction) {
    Tensor t1({4, 4}, Device::CPU);
    float* original_ptr = t1.data();

    Tensor t2(std::move(t1));
    EXPECT_EQ(t2.data(), original_ptr);
    EXPECT_TRUE(t2.is_valid());
    EXPECT_FALSE(t1.is_valid());  // Moved-from state
}

TEST(TensorTest, MoveAssignment) {
    Tensor t1({8}, Device::CPU);
    float* original_ptr = t1.data();

    Tensor t2;
    t2 = std::move(t1);
    EXPECT_EQ(t2.data(), original_ptr);
    EXPECT_TRUE(t2.is_valid());
    EXPECT_FALSE(t1.is_valid());
}

// ─── Reshape ────────────────────────────────────────────────────────────────

TEST(TensorTest, Reshape) {
    Tensor t({2, 3}, Device::CPU);
    EXPECT_EQ(t.numel(), 6u);

    t.reshape({6});
    EXPECT_EQ(t.ndim(), 1);
    EXPECT_EQ(t.size(0), 6);
    EXPECT_EQ(t.numel(), 6u);

    t.reshape({3, 2});
    EXPECT_EQ(t.ndim(), 2);
    EXPECT_EQ(t.size(0), 3);
    EXPECT_EQ(t.size(1), 2);
}

// ─── Large Tensors ──────────────────────────────────────────────────────────

TEST(TensorTest, LargeTensorGPU) {
    // Allocate a 1M element tensor on GPU
    const int n = 1024 * 1024;
    Tensor t({n}, Device::CUDA);
    EXPECT_TRUE(t.is_valid());
    EXPECT_EQ(t.numel(), static_cast<size_t>(n));
    EXPECT_EQ(t.nbytes(), n * sizeof(float));
}
