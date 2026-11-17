
#pragma once

#include "core/dtype.h"
#include "core/device.h"

#include <cstddef>
#include <memory>
#include <vector>
#include <stdexcept>

namespace core {

class Tensor {

public:

    Tensor() = default;

    Tensor(
        std::vector<size_t> shape,
        DType dtype = DType::FP32,
        Device device = Device::cpu()
    );

    size_t numel() const;

    size_t nbytes() const;

    const std::vector<size_t>& shape() const;

    const std::vector<size_t>& strides() const;

    DType dtype() const;

    Device device() const;

    float* data();

    const float* data() const;

    bool empty() const;

private:

    std::vector<size_t> shape_;

    std::vector<size_t> strides_;

    DType dtype_ = DType::FP32;

    Device device_ = Device::cpu();

    std::shared_ptr<void> storage_;

    size_t numel_ = 0;

    void allocate();

    void calculate_strides();
};

}
