
#pragma once

#include <string>
#include "core/tensor.h"

namespace core {

bool save_tensor_binary(
    const Tensor& tensor,
    const std::string& path
);

}
