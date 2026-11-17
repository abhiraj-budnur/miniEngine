
#pragma once

#include "core/tensor.h"

namespace ops {

core::Tensor swiglu(
    const core::Tensor& gate,
    const core::Tensor& up
);

}
