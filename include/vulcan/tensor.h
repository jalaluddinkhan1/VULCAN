#pragma once

/// @file tensor.h
/// @brief VULCAN Tensor — GPU-aware tensor with RAII memory management.
///
/// Core data structure for the inference engine. Manages raw float*
/// pointers on both CPU and CUDA devices with move-only semantics
/// to prevent double-free.

#include <vector>
#include <string>
#include <cstddef>
#include <cassert>
#include <memory>
#include <stdexcept>
#include <iostream>

namespace vulcan {

/// Device on which tensor data resides.
enum class Device {
    CPU,
    CUDA
};

/// @class Tensor
/// @brief A multi-dimensional tensor with raw pointer GPU memory management.
///
/// Design decisions:
///   - Raw float* instead of std::vector for GPU compatibility
///   - Move-only (deleted copy ctor/assignment) to prevent double cudaFree
///   - RAII: cudaMalloc on construction, cudaFree on destruction
///   - Shape stored as std::vector<int> for flexibility
class Tensor {
public:
    // ─── Construction / Destruction ─────────────────────────────────────

    /// Default constructor — creates an empty tensor.
    Tensor();

    /// Construct a tensor with given shape on the specified device.
    /// Memory is allocated but NOT initialized.
    /// @param shape  Dimensions (e.g., {batch, seq_len, hidden_dim})
    /// @param device Target device (CPU or CUDA)
    explicit Tensor(const std::vector<int>& shape, Device device = Device::CPU);

    /// Destructor — frees CPU or GPU memory.
    ~Tensor();

    // ─── Move Semantics (No Copy) ───────────────────────────────────────

    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // ─── Device Transfer ────────────────────────────────────────────────

    /// Copy tensor data to the specified device, returning a new Tensor.
    /// @param target Target device
    /// @return New Tensor on the target device
    Tensor to(Device target) const;

    /// Load data from a host (CPU) float array into this tensor.
    /// @param host_data Pointer to host data (must have numel() floats)
    void from_host(const float* host_data);

    /// Copy tensor data to a host (CPU) float array.
    /// @param host_data Pointer to host buffer (must have space for numel() floats)
    void to_host(float* host_data) const;

    // ─── Accessors ──────────────────────────────────────────────────────

    /// Raw pointer to underlying data.
    float* data();
    const float* data() const;

    /// Shape of the tensor.
    const std::vector<int>& shape() const;

    /// Number of dimensions.
    int ndim() const;

    /// Size along a specific dimension.
    int size(int dim) const;

    /// Total number of elements.
    size_t numel() const;

    /// Total size in bytes.
    size_t nbytes() const;

    /// Device where data resides.
    Device device() const;

    /// Whether the tensor has allocated memory.
    bool is_valid() const;

    // ─── Utility ────────────────────────────────────────────────────────

    /// Print tensor metadata to stdout (shape, device, data pointer).
    void print_info(const std::string& name = "") const;

    /// Reshape tensor (must preserve numel).
    /// @param new_shape New dimensions
    void reshape(const std::vector<int>& new_shape);

private:
    float*           data_;     ///< Raw pointer to data (CPU or GPU)
    std::vector<int> shape_;    ///< Tensor dimensions
    Device           device_;   ///< Current device
    size_t           numel_;    ///< Cached element count

    /// Allocate memory on the current device.
    void allocate();

    /// Free memory on the current device.
    void deallocate();
};

} // namespace vulcan
