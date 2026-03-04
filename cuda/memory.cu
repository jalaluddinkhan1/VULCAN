/// @file memory.cu
/// @brief GPUBuffer and MemoryManager — CUDA memory management implementation.
///
/// GPUBuffer: RAII wrapper for cudaMalloc/cudaFree with move semantics.
/// MemoryManager: Paged allocator for KV cache — fixed-size blocks with
///   block table mapping and CPU swap support.


#include "cuda/memory.h"
#include "cuda/utils.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <cstring>

namespace vulcan {
namespace cuda {

// ═══════════════════════════════════════════════════════════════════════════
//  GPUBuffer
// ═══════════════════════════════════════════════════════════════════════════

GPUBuffer::GPUBuffer()
    : data_(nullptr), size_(0) {}

GPUBuffer::GPUBuffer(size_t size_bytes)
    : data_(nullptr), size_(size_bytes) {
    if (size_bytes > 0) {
        CUDA_CHECK(cudaMalloc(&data_, size_bytes));
    }
}

GPUBuffer::~GPUBuffer() {
    if (data_) {
        cudaFree(data_);
        data_ = nullptr;
    }
}

GPUBuffer::GPUBuffer(GPUBuffer&& other) noexcept
    : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
}

GPUBuffer& GPUBuffer::operator=(GPUBuffer&& other) noexcept {
    if (this != &other) {
        if (data_) cudaFree(data_);
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

void* GPUBuffer::data() { return data_; }
const void* GPUBuffer::data() const { return data_; }
size_t GPUBuffer::size() const { return size_; }
bool GPUBuffer::is_valid() const { return data_ != nullptr; }

void* GPUBuffer::release() {
    void* ptr = data_;
    data_ = nullptr;
    size_ = 0;
    return ptr;
}

// ═══════════════════════════════════════════════════════════════════════════
//  MemoryManager — Paged GPU Memory Allocator
// ═══════════════════════════════════════════════════════════════════════════

MemoryManager::MemoryManager(size_t page_size, int max_pages)
    : page_size_(page_size), max_pages_(max_pages) {
    // Pre-allocate the entire block pool as a single contiguous allocation
    // This eliminates fragmentation — the core design principle from ADR-001.
    blocks_.reserve(max_pages);

    // Allocate one large GPU buffer and carve it into pages
    size_t total_bytes = page_size * max_pages;
    void* pool = nullptr;
    cudaError_t err = cudaMalloc(&pool, total_bytes);

    if (err != cudaSuccess) {
        // Graceful degradation: try allocating fewer pages
        int reduced = max_pages / 2;
        while (reduced > 0) {
            total_bytes = page_size * reduced;
            err = cudaMalloc(&pool, total_bytes);
            if (err == cudaSuccess) {
                max_pages_ = reduced;
                std::cerr << "[VULCAN] Warning: Reduced page pool to "
                          << reduced << " pages ("
                          << (total_bytes / (1024*1024)) << " MB)" << std::endl;
                break;
            }
            reduced /= 2;
        }
        if (err != cudaSuccess) {
            std::cerr << "[VULCAN] Error: Cannot allocate page pool." << std::endl;
            max_pages_ = 0;
            return;
        }
    }

    // Initialize block metadata
    uint8_t* base = static_cast<uint8_t*>(pool);
    for (int i = 0; i < max_pages_; ++i) {
        MemoryBlock block;
        block.ptr      = base + i * page_size;
        block.size     = page_size;
        block.in_use   = false;
        block.block_id = i;
        blocks_.push_back(block);
    }
}

MemoryManager::~MemoryManager() {
    // Free the entire pool (first block's pointer is the pool base)
    if (!blocks_.empty() && blocks_[0].ptr) {
        cudaFree(blocks_[0].ptr);
    }
    // Also free any CPU swap buffers
    cpu_swap_.clear();
}

int MemoryManager::allocate_block() {
    for (auto& block : blocks_) {
        if (!block.in_use) {
            block.in_use = true;
            return block.block_id;
        }
    }
    return -1;  // No free blocks
}

void MemoryManager::free_block(int block_id) {
    if (block_id >= 0 && block_id < static_cast<int>(blocks_.size())) {
        blocks_[block_id].in_use = false;
        // Also remove from CPU swap if it was swapped
        cpu_swap_.erase(block_id);
    }
}

void* MemoryManager::get_block_ptr(int block_id) const {
    if (block_id >= 0 && block_id < static_cast<int>(blocks_.size())) {
        return blocks_[block_id].ptr;
    }
    return nullptr;
}

bool MemoryManager::swap_to_cpu(int block_id) {
    if (block_id < 0 || block_id >= static_cast<int>(blocks_.size())) return false;
    if (!blocks_[block_id].in_use) return false;

    // Allocate CPU-side storage
    std::vector<uint8_t> cpu_buf(page_size_);

    // Copy GPU → CPU
    cudaError_t err = cudaMemcpy(cpu_buf.data(), blocks_[block_id].ptr,
                                  page_size_, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return false;

    cpu_swap_[block_id] = std::move(cpu_buf);

    // Mark GPU block as free (data is on CPU now)
    blocks_[block_id].in_use = false;
    return true;
}

bool MemoryManager::swap_to_gpu(int block_id) {
    if (block_id < 0 || block_id >= static_cast<int>(blocks_.size())) return false;

    auto it = cpu_swap_.find(block_id);
    if (it == cpu_swap_.end()) return false;  // Nothing to swap in

    // Re-allocate the GPU block
    blocks_[block_id].in_use = true;

    // Copy CPU → GPU
    cudaError_t err = cudaMemcpy(blocks_[block_id].ptr, it->second.data(),
                                  page_size_, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        blocks_[block_id].in_use = false;
        return false;
    }

    // Remove CPU copy
    cpu_swap_.erase(it);
    return true;
}

int MemoryManager::blocks_in_use() const {
    int count = 0;
    for (const auto& b : blocks_) {
        if (b.in_use) ++count;
    }
    return count;
}

int MemoryManager::blocks_available() const {
    return max_pages_ - blocks_in_use();
}

size_t MemoryManager::total_allocated_bytes() const {
    return static_cast<size_t>(blocks_in_use()) * page_size_;
}

} // namespace cuda
} // namespace vulcan
