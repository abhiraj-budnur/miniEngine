
#pragma once

#include "core/tensor.h"

namespace ops {

core::Tensor matmul(
    const core::Tensor& A,
    const core::Tensor& B
);

}
