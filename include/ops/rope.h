
#pragma once

#include "core/tensor.h"

namespace ops {

void apply_rope(
    core::Tensor& input,
    int position,
    float theta = 1000000.0f
);

}
