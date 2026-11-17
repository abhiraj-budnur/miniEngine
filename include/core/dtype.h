
#pragma once

#include <cstddef>
#include <cstdint>

namespace core {

enum class DType {
    FP32,
    FP16,
    BF16,
    INT8,
    INT32
};

inline size_t dtype_size(DType dtype) {

    switch (dtype) {

        case DType::FP32:
            return sizeof(float);

        case DType::FP16:
        case DType::BF16:
            return sizeof(uint16_t);

        case DType::INT8:
            return sizeof(int8_t);

        case DType::INT32:
            return sizeof(int32_t);
    }

    return 0;
}

}
