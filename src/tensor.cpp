/// @file tensor.cpp
/// @brief VULCAN Tensor — CPU-side implementation.

#include "vulcan/tensor.h"
#include "cuda/utils.h"
#include <cuda_runtime.h>
#include <numeric>
#include <algorithm>
#include <cstring>

namespace vulcan {

// ─── Construction / Destruction ─────────────────────────────────────────────

Tensor::Tensor()
    : data_(nullptr), shape_(), device_(Device::CPU), numel_(0) {}

Tensor::Tensor(const std::vector<int>& shape, Device device)
    : data_(nullptr), shape_(shape), device_(device), numel_(0) {
    // Compute total element count
    numel_ = 1;
    for (int dim : shape_) {
        assert(dim > 0 && "Tensor dimensions must be positive");
        numel_ *= static_cast<size_t>(dim);
    }
    allocate();
}

Tensor::~Tensor() {
    deallocate();
}

// ─── Move Semantics ─────────────────────────────────────────────────────────

Tensor::Tensor(Tensor&& other) noexcept
    : data_(other.data_),
      shape_(std::move(other.shape_)),
      device_(other.device_),
      numel_(other.numel_) {
    other.data_  = nullptr;
    other.numel_ = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        deallocate();
        data_   = other.data_;
        shape_  = std::move(other.shape_);
        device_ = other.device_;
        numel_  = other.numel_;

        other.data_  = nullptr;
        other.numel_ = 0;
    }
    return *this;
}

// ─── Device Transfer ────────────────────────────────────────────────────────

Tensor Tensor::to(Device target) const {
    if (target == device_) {
        // Same device: create a copy
        Tensor copy(shape_, device_);
        if (device_ == Device::CPU) {
            std::memcpy(copy.data_, data_, nbytes());
        } else {
            CUDA_CHECK(cudaMemcpy(copy.data_, data_, nbytes(),
                                  cudaMemcpyDeviceToDevice));
        }
        return copy;
    }

    Tensor result(shape_, target);

    if (device_ == Device::CPU && target == Device::CUDA) {
        // CPU → GPU
        CUDA_CHECK(cudaMemcpy(result.data_, data_, nbytes(),
                              cudaMemcpyHostToDevice));
    } else if (device_ == Device::CUDA && target == Device::CPU) {
        // GPU → CPU
        CUDA_CHECK(cudaMemcpy(result.data_, data_, nbytes(),
                              cudaMemcpyDeviceToHost));
    }

    return result;
}

void Tensor::from_host(const float* host_data) {
    assert(data_ && "Tensor must be allocated before loading data");
    assert(host_data && "Source data pointer must not be null");

    if (device_ == Device::CPU) {
        std::memcpy(data_, host_data, nbytes());
    } else {
        CUDA_CHECK(cudaMemcpy(data_, host_data, nbytes(),
                              cudaMemcpyHostToDevice));
    }
}

void Tensor::to_host(float* host_data) const {
    assert(data_ && "Tensor must be allocated before reading data");
    assert(host_data && "Destination buffer must not be null");

    if (device_ == Device::CPU) {
        std::memcpy(host_data, data_, nbytes());
    } else {
        CUDA_CHECK(cudaMemcpy(host_data, data_, nbytes(),
                              cudaMemcpyDeviceToHost));
    }
}

// ─── Accessors ──────────────────────────────────────────────────────────────

float* Tensor::data() { return data_; }
const float* Tensor::data() const { return data_; }

const std::vector<int>& Tensor::shape() const { return shape_; }

int Tensor::ndim() const { return static_cast<int>(shape_.size()); }

int Tensor::size(int dim) const {
    assert(dim >= 0 && dim < ndim() && "Dimension out of range");
    return shape_[dim];
}

size_t Tensor::numel() const { return numel_; }

size_t Tensor::nbytes() const { return numel_ * sizeof(float); }

Device Tensor::device() const { return device_; }

bool Tensor::is_valid() const { return data_ != nullptr && numel_ > 0; }

// ─── Utility ────────────────────────────────────────────────────────────────

void Tensor::print_info(const std::string& name) const {
    if (!name.empty()) {
        std::cout << "Tensor '" << name << "': ";
    } else {
        std::cout << "Tensor: ";
    }

    std::cout << "shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << shape_[i];
    }
    std::cout << "], "
              << "numel=" << numel_ << ", "
              << "device=" << (device_ == Device::CUDA ? "CUDA" : "CPU") << ", "
              << "data=" << static_cast<void*>(data_)
              << std::endl;
}

void Tensor::reshape(const std::vector<int>& new_shape) {
    size_t new_numel = 1;
    for (int dim : new_shape) {
        assert(dim > 0 && "Reshape dimensions must be positive");
        new_numel *= static_cast<size_t>(dim);
    }
    assert(new_numel == numel_ && "Reshape must preserve total element count");
    shape_ = new_shape;
}

// ─── Private Helpers ────────────────────────────────────────────────────────

void Tensor::allocate() {
    if (numel_ == 0) return;

    if (device_ == Device::CPU) {
        data_ = new float[numel_];
    } else {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), nbytes()));
    }
}

void Tensor::deallocate() {
    if (!data_) return;

    if (device_ == Device::CPU) {
        delete[] data_;
    } else {
        cudaFree(data_);  // Don't check error in destructor
    }

    data_  = nullptr;
    numel_ = 0;
}

} // namespace vulcan
