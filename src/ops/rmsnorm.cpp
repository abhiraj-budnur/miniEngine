
#include "ops/rmsnorm.h"

#include <cmath>
#include <stdexcept>

namespace ops {

core::Tensor rmsnorm(
    const core::Tensor& input,
    const core::Tensor& weight,
    float epsilon
) {

    if (input.shape().empty()) {
        throw std::invalid_argument("Invalid input");
    }

    size_t hidden = input.shape().back();

    if (weight.numel() != hidden) {
        throw std::invalid_argument(
            "RMSNorm weight dimension mismatch"
        );
    }

    if (epsilon <= 0.0f) {
        throw std::invalid_argument(
            "Epsilon must be positive"
        );
    }

    core::Tensor output(input.shape());

    size_t rows = input.numel() / hidden;

    for (size_t row = 0; row < rows; row++) {

        const float* x =
            input.data() + row * hidden;

        float* y =
            output.data() + row * hidden;

        float square_sum = 0.0f;

        for (size_t i = 0; i < hidden; i++) {
            square_sum += x[i] * x[i];
        }

        float mean_square =
            square_sum / static_cast<float>(hidden);

        float inverse_rms =
            1.0f / std::sqrt(mean_square + epsilon);

        for (size_t i = 0; i < hidden; i++) {

            y[i] =
                x[i] *
                inverse_rms *
                weight.data()[i];
        }
    }

    return output;
}

}
