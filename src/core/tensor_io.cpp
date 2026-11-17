
#include "core/tensor_io.h"

#include <fstream>
#include <cstdint>

namespace core {

bool save_tensor_binary(
    const Tensor& tensor,
    const std::string& path
) {
    if (tensor.empty()) {
        return false;
    }

    if (tensor.dtype() != DType::FP32) {
        return false;
    }

    const float* ptr = tensor.data();

    if (ptr == nullptr) {
        return false;
    }

    std::ofstream file(
        path,
        std::ios::binary | std::ios::trunc
    );

    if (!file.is_open()) {
        return false;
    }

    const std::size_t bytes =
        tensor.numel() * sizeof(float);

    file.write(
        reinterpret_cast<const char*>(ptr),
        static_cast<std::streamsize>(bytes)
    );

    return file.good();
}

}
