
#pragma once

#include "core/tensor.h"

namespace ops {

core::Tensor rmsnorm(
    const core::Tensor& input,
    const core::Tensor& weight,
    float epsilon = 1e-6f
);

}
