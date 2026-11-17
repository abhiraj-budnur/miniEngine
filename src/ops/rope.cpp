
#include "ops/rope.h"

#include <cmath>
#include <stdexcept>

namespace ops {

void apply_rope(
    core::Tensor& input,
    int position,
    float theta
) {

    if (input.shape().empty()) {
        throw std::invalid_argument("Invalid RoPE input");
    }

    if (position < 0 || theta <= 0.0f) {
        throw std::invalid_argument(
            "Invalid RoPE position or theta"
        );
    }

    size_t head_dim = input.shape().back();

    if (head_dim % 2 != 0) {
        throw std::invalid_argument(
            "RoPE requires an even head dimension"
        );
    }

    size_t rows = input.numel() / head_dim;

    for (size_t row = 0; row < rows; row++) {

        float* x =
            input.data() + row * head_dim;

        for (size_t i = 0; i < head_dim; i += 2) {

            float exponent =
                static_cast<float>(i) /
                static_cast<float>(head_dim);

            float frequency =
                1.0f / std::pow(theta, exponent);

            float angle =
                static_cast<float>(position) * frequency;

            float cos_value = std::cos(angle);
            float sin_value = std::sin(angle);

            float x0 = x[i];
            float x1 = x[i + 1];

            x[i] =
                x0 * cos_value -
                x1 * sin_value;

            x[i + 1] =
                x0 * sin_value +
                x1 * cos_value;
        }
    }
}

}
