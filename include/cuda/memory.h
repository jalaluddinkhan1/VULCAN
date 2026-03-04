#pragma once

/// @file memory.h
/// @brief CUDA Memory Manager — RAII GPU buffer and paged allocator.
///
/// Provides safe GPU memory management:
///   - GPUBuffer: RAII wrapper for cudaMalloc/cudaFree
///   - MemoryManager: Paged allocation for KV cache

#include <cstddef>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace vulcan {
namespace cuda {

/// @class GPUBuffer
/// @brief RAII wrapper for GPU memory allocation.
///
/// Allocates GPU memory on construction, frees on destruction.
/// Move-only to prevent double-free.
class GPUBuffer {
public:
    /// Default constructor — no allocation.
    GPUBuffer();

    /// Allocate `size_bytes` of GPU memory.
    explicit GPUBuffer(size_t size_bytes);

    /// Destructor — calls cudaFree.
    ~GPUBuffer();

    // Move only
    GPUBuffer(GPUBuffer&& other) noexcept;
    GPUBuffer& operator=(GPUBuffer&& other) noexcept;
    GPUBuffer(const GPUBuffer&) = delete;
    GPUBuffer& operator=(const GPUBuffer&) = delete;

    /// Get raw device pointer.
    void* data();
    const void* data() const;

    /// Get typed device pointer.
    template <typename T>
    T* as() { return static_cast<T*>(data_); }

    template <typename T>
    const T* as() const { return static_cast<const T*>(data_); }

    /// Allocated size in bytes.
    size_t size() const;

    /// Whether memory is allocated.
    bool is_valid() const;

    /// Release ownership and return pointer (caller must cudaFree).
    void* release();

private:
    void*  data_;
    size_t size_;
};

// ─── Paged Memory Manager ──────────────────────────────────────────────────

/// Default page size: 16 KB
constexpr size_t DEFAULT_PAGE_SIZE = 16 * 1024;

/// @struct MemoryBlock
/// @brief Represents a single page of GPU memory.
struct MemoryBlock {
    void*    ptr;        ///< Device pointer to the block
    size_t   size;       ///< Block size in bytes
    bool     in_use;     ///< Whether this block is currently allocated
    int      block_id;   ///< Unique identifier
};

/// @class MemoryManager
/// @brief Paged GPU memory allocator for KV cache management.
///
/// Allocates GPU memory in fixed-size pages. Maintains a block table
/// mapping logical blocks to physical GPU blocks. Supports swapping
/// blocks to/from CPU RAM when GPU VRAM is exhausted.
///

class MemoryManager {
public:
    /// @param page_size     Size of each memory page in bytes
    /// @param max_pages     Maximum number of pages to allocate
    explicit MemoryManager(size_t page_size = DEFAULT_PAGE_SIZE,
                           int max_pages = 1024);
    ~MemoryManager();

    /// Allocate a logical block, returning its block ID.
    /// @return Block ID, or -1 if no pages available
    int allocate_block();

    /// Free a previously allocated block.
    /// @param block_id Block to free
    void free_block(int block_id);

    /// Get the device pointer for a given block.
    /// @param block_id Block to look up
    /// @return Device pointer, or nullptr if invalid
    void* get_block_ptr(int block_id) const;

    /// Swap a GPU block to CPU RAM (when VRAM is full).
    /// @param block_id Block to swap out
    /// @return true on success
    bool swap_to_cpu(int block_id);

    /// Swap a block back from CPU RAM to GPU.
    /// @param block_id Block to swap in
    /// @return true on success
    bool swap_to_gpu(int block_id);

    /// Get usage statistics.
    int blocks_in_use() const;
    int blocks_available() const;
    size_t total_allocated_bytes() const;

private:
    size_t                    page_size_;
    int                       max_pages_;
    std::vector<MemoryBlock>  blocks_;
    std::unordered_map<int, std::vector<uint8_t>> cpu_swap_; ///< CPU-side swap storage
};

} // namespace cuda
} // namespace vulcan
