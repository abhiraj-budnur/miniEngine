
#include "core/tensor.h"

#include <limits>
#include <new>
#include <numeric>

namespace core {

Tensor::Tensor(
    std::vector<size_t> shape,
    DType dtype,
    Device device
)
    : shape_(std::move(shape)),
      dtype_(dtype),
      device_(device) {

    if (shape_.empty()) {
        throw std::invalid_argument(
            "Tensor shape cannot be empty"
        );
    }

    if (dtype_ != DType::FP32) {
        throw std::runtime_error(
            "Only FP32 tensors are currently supported"
        );
    }

    if (device_.type != DeviceType::CPU) {
        throw std::runtime_error(
            "Metal tensor allocation is not implemented yet"
        );
    }

    numel_ = 1;

    for (size_t dimension : shape_) {

        if (dimension == 0) {
            numel_ = 0;
            break;
        }

        if (numel_ >
            std::numeric_limits<size_t>::max() / dimension) {

            throw std::overflow_error(
                "Tensor dimensions overflow"
            );
        }

        numel_ *= dimension;
    }

    calculate_strides();
    allocate();
}

void Tensor::calculate_strides() {

    strides_.resize(shape_.size());

    size_t stride = 1;

    for (int i = static_cast<int>(shape_.size()) - 1;
         i >= 0;
         --i) {

        strides_[i] = stride;

        stride *= shape_[i];
    }
}

void Tensor::allocate() {

    if (numel_ == 0) {
        return;
    }

    size_t bytes = numel_ * sizeof(float);

    void* ptr = ::operator new(bytes);

    storage_ = std::shared_ptr<void>(
        ptr,
        [](void* p) {
            ::operator delete(p);
        }
    );
}

size_t Tensor::numel() const {
    return numel_;
}

size_t Tensor::nbytes() const {
    return numel_ * dtype_size(dtype_);
}

const std::vector<size_t>& Tensor::shape() const {
    return shape_;
}

const std::vector<size_t>& Tensor::strides() const {
    return strides_;
}

DType Tensor::dtype() const {
    return dtype_;
}

Device Tensor::device() const {
    return device_;
}

float* Tensor::data() {

    return static_cast<float*>(storage_.get());
}

const float* Tensor::data() const {

    return static_cast<const float*>(
        storage_.get()
    );
}

bool Tensor::empty() const {
    return numel_ == 0;
}

}
