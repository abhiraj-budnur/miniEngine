
#include "ops/swiglu.h"

#include <cmath>
#include <stdexcept>

namespace ops {

core::Tensor swiglu(
    const core::Tensor& gate,
    const core::Tensor& up
) {

    if (gate.shape() != up.shape()) {
        throw std::invalid_argument(
            "SwiGLU input dimensions must match"
        );
    }

    core::Tensor output(gate.shape());

    for (size_t i = 0; i < gate.numel(); i++) {

        float x = gate.data()[i];

        float silu =
            x / (1.0f + std::exp(-x));

        output.data()[i] =
            silu * up.data()[i];
    }

    return output;
}

}
