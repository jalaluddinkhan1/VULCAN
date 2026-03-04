/// @file test_memory.cpp
/// @brief Tests for GPUBuffer, MemoryManager, and KVCache.
///
/// Validates: RAII memory management, paged allocation, GPU↔CPU swap,
///            KV cache append/get/reset.

#include <gtest/gtest.h>
#include "cuda/memory.h"
#include "cuda/utils.h"
#include "vulcan/kv_cache.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstring>

using namespace vulcan::cuda;

// ═══════════════════════════════════════════════════════════════════════════
//  GPUBuffer Tests
// ═══════════════════════════════════════════════════════════════════════════

TEST(GPUBufferTest, DefaultConstruction) {
    GPUBuffer buf;
    EXPECT_FALSE(buf.is_valid());
    EXPECT_EQ(buf.size(), 0u);
    EXPECT_EQ(buf.data(), nullptr);
}

TEST(GPUBufferTest, Allocation) {
    size_t size = 1024 * sizeof(float);
    GPUBuffer buf(size);
    EXPECT_TRUE(buf.is_valid());
    EXPECT_EQ(buf.size(), size);
    EXPECT_NE(buf.data(), nullptr);
}

TEST(GPUBufferTest, ZeroSizeAllocation) {
    GPUBuffer buf(0);
    EXPECT_FALSE(buf.is_valid());
    EXPECT_EQ(buf.size(), 0u);
}

TEST(GPUBufferTest, MoveConstruction) {
    GPUBuffer original(4096);
    void* orig_ptr = original.data();
    size_t orig_size = original.size();

    GPUBuffer moved(std::move(original));
    EXPECT_EQ(moved.data(), orig_ptr);
    EXPECT_EQ(moved.size(), orig_size);
    EXPECT_TRUE(moved.is_valid());

    // Original should be empty
    EXPECT_FALSE(original.is_valid());
    EXPECT_EQ(original.data(), nullptr);
    EXPECT_EQ(original.size(), 0u);
}

TEST(GPUBufferTest, MoveAssignment) {
    GPUBuffer buf1(1024);
    GPUBuffer buf2(2048);
    void* ptr2 = buf2.data();

    buf1 = std::move(buf2);
    EXPECT_EQ(buf1.data(), ptr2);
    EXPECT_EQ(buf1.size(), 2048u);
    EXPECT_FALSE(buf2.is_valid());
}

TEST(GPUBufferTest, Release) {
    GPUBuffer buf(1024);
    void* ptr = buf.data();
    void* released = buf.release();
    EXPECT_EQ(released, ptr);
    EXPECT_FALSE(buf.is_valid());

    // Manual cleanup since we released ownership
    cudaFree(released);
}

TEST(GPUBufferTest, TypedAccess) {
    GPUBuffer buf(256 * sizeof(float));
    float* fptr = buf.as<float>();
    EXPECT_NE(fptr, nullptr);

    // Write and read back to verify the pointer is usable
    std::vector<float> h_data(256, 3.14f);
    cudaMemcpy(fptr, h_data.data(), 256 * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<float> h_out(256, 0.0f);
    cudaMemcpy(h_out.data(), fptr, 256 * sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_FLOAT_EQ(h_out[0], 3.14f);
    EXPECT_FLOAT_EQ(h_out[255], 3.14f);
}

// ═══════════════════════════════════════════════════════════════════════════
//  MemoryManager Tests
// ═══════════════════════════════════════════════════════════════════════════

TEST(MemoryManagerTest, Construction) {
    MemoryManager mgr(4096, 16);  // 16 pages of 4KB
    EXPECT_EQ(mgr.blocks_in_use(), 0);
    EXPECT_EQ(mgr.blocks_available(), 16);
    EXPECT_EQ(mgr.total_allocated_bytes(), 0u);
}

TEST(MemoryManagerTest, AllocateAndFree) {
    MemoryManager mgr(4096, 8);

    int id = mgr.allocate_block();
    EXPECT_GE(id, 0);
    EXPECT_EQ(mgr.blocks_in_use(), 1);
    EXPECT_EQ(mgr.blocks_available(), 7);

    void* ptr = mgr.get_block_ptr(id);
    EXPECT_NE(ptr, nullptr);

    mgr.free_block(id);
    EXPECT_EQ(mgr.blocks_in_use(), 0);
    EXPECT_EQ(mgr.blocks_available(), 8);
}

TEST(MemoryManagerTest, AllocateAll) {
    MemoryManager mgr(1024, 4);

    std::vector<int> ids;
    for (int i = 0; i < 4; ++i) {
        int id = mgr.allocate_block();
        EXPECT_GE(id, 0);
        ids.push_back(id);
    }

    // All blocks used
    EXPECT_EQ(mgr.blocks_in_use(), 4);
    EXPECT_EQ(mgr.blocks_available(), 0);

    // Next allocation should fail
    int fail_id = mgr.allocate_block();
    EXPECT_EQ(fail_id, -1);

    // Free one, allocate again
    mgr.free_block(ids[0]);
    EXPECT_EQ(mgr.blocks_available(), 1);

    int new_id = mgr.allocate_block();
    EXPECT_GE(new_id, 0);
}

TEST(MemoryManagerTest, BlockPointersPersist) {
    MemoryManager mgr(4096, 8);

    int id1 = mgr.allocate_block();
    int id2 = mgr.allocate_block();

    void* ptr1 = mgr.get_block_ptr(id1);
    void* ptr2 = mgr.get_block_ptr(id2);

    EXPECT_NE(ptr1, ptr2);
    EXPECT_NE(ptr1, nullptr);
    EXPECT_NE(ptr2, nullptr);

    // Pointers should be page_size apart (contiguous pool)
    ptrdiff_t diff = static_cast<uint8_t*>(ptr2) - static_cast<uint8_t*>(ptr1);
    EXPECT_EQ(std::abs(diff), 4096);

    mgr.free_block(id1);
    mgr.free_block(id2);
}

TEST(MemoryManagerTest, SwapToCPUAndBack) {
    MemoryManager mgr(1024, 4);

    int id = mgr.allocate_block();
    void* ptr = mgr.get_block_ptr(id);

    // Write known data to the GPU block
    std::vector<uint8_t> test_data(1024);
    for (int i = 0; i < 1024; ++i) test_data[i] = static_cast<uint8_t>(i % 256);
    cudaMemcpy(ptr, test_data.data(), 1024, cudaMemcpyHostToDevice);

    // Swap to CPU
    bool swap_out = mgr.swap_to_cpu(id);
    EXPECT_TRUE(swap_out);
    EXPECT_EQ(mgr.blocks_in_use(), 0);  // GPU block freed

    // Block is free on GPU now — can be reused
    int temp = mgr.allocate_block();
    EXPECT_EQ(temp, id);  // Same block should be reused
    mgr.free_block(temp);

    // Swap back to GPU
    bool swap_in = mgr.swap_to_gpu(id);
    EXPECT_TRUE(swap_in);
    EXPECT_EQ(mgr.blocks_in_use(), 1);

    // Verify data integrity
    std::vector<uint8_t> readback(1024);
    cudaMemcpy(readback.data(), mgr.get_block_ptr(id), 1024, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 1024; ++i) {
        EXPECT_EQ(readback[i], test_data[i]) << "Data corruption at byte " << i;
    }

    mgr.free_block(id);
}

TEST(MemoryManagerTest, InvalidBlockId) {
    MemoryManager mgr(1024, 4);

    EXPECT_EQ(mgr.get_block_ptr(-1), nullptr);
    EXPECT_EQ(mgr.get_block_ptr(999), nullptr);
    EXPECT_FALSE(mgr.swap_to_cpu(-1));
    EXPECT_FALSE(mgr.swap_to_gpu(999));
}

// ═══════════════════════════════════════════════════════════════════════════
//  KVCache Tests
// ═══════════════════════════════════════════════════════════════════════════

TEST(KVCacheTest, Construction) {
    vulcan::KVCache cache(4, 128, 64, 2);  // 4 heads, 128 max_seq, 64 head_dim, 2 layers
    EXPECT_EQ(cache.current_length(), 0);
    EXPECT_EQ(cache.max_seq_len(), 128);
    EXPECT_GT(cache.memory_usage(), 0u);
}

TEST(KVCacheTest, AppendAndGet) {
    const int num_heads = 2, max_seq = 32, head_dim = 4, num_layers = 1;
    vulcan::KVCache cache(num_heads, max_seq, head_dim, num_layers);

    // Create a single token's K and V: [1, num_heads * head_dim]
    int kv_dim = num_heads * head_dim;
    std::vector<float> h_k(kv_dim), h_v(kv_dim);
    for (int i = 0; i < kv_dim; ++i) {
        h_k[i] = static_cast<float>(i + 1);
        h_v[i] = static_cast<float>(i + 100);
    }

    // Upload to GPU
    float *d_k, *d_v;
    cudaMalloc(&d_k, kv_dim * sizeof(float));
    cudaMalloc(&d_v, kv_dim * sizeof(float));
    cudaMemcpy(d_k, h_k.data(), kv_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), kv_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Append token at position 0
    cache.append(0, d_k, d_v, 0, 1);
    EXPECT_EQ(cache.current_length(), 1);

    // Get cached pointers
    const float *k_out, *v_out;
    cache.get(0, &k_out, &v_out);
    EXPECT_NE(k_out, nullptr);
    EXPECT_NE(v_out, nullptr);

    cudaFree(d_k);
    cudaFree(d_v);
}

TEST(KVCacheTest, Reset) {
    vulcan::KVCache cache(4, 64, 32, 1);
    cache.set_length(42);
    EXPECT_EQ(cache.current_length(), 42);

    cache.reset();
    EXPECT_EQ(cache.current_length(), 0);
}

TEST(KVCacheTest, MoveSemantics) {
    vulcan::KVCache cache1(4, 64, 32, 2);
    cache1.set_length(10);

    vulcan::KVCache cache2(std::move(cache1));
    EXPECT_EQ(cache2.current_length(), 10);
    EXPECT_EQ(cache2.max_seq_len(), 64);

    // Original should be empty
    EXPECT_EQ(cache1.current_length(), 0);
}

TEST(KVCacheTest, MemoryUsageCalculation) {
    const int heads = 8, max_seq = 2048, head_dim = 128, layers = 32;
    vulcan::KVCache cache(heads, max_seq, head_dim, layers);

    // Expected: 2 (K+V) * 32 layers * 8 heads * 2048 seq * 128 dim * 4 bytes
    size_t expected = 2ULL * layers * heads * max_seq * head_dim * sizeof(float);
    EXPECT_EQ(cache.memory_usage(), expected);
}
