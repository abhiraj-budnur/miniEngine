
#include "ops/softmax.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ops {

core::Tensor softmax(
    const core::Tensor& input
) {

    if (input.shape().empty()) {
        throw std::invalid_argument(
            "Invalid softmax input"
        );
    }

    size_t cols = input.shape().back();

    if (cols == 0) {
        return core::Tensor(input.shape());
    }

    size_t rows = input.numel() / cols;

    core::Tensor output(input.shape());

    for (size_t row = 0; row < rows; row++) {

        const float* x =
            input.data() + row * cols;

        float* y =
            output.data() + row * cols;

        float maximum = x[0];

        for (size_t i = 1; i < cols; i++) {
            maximum = std::max(maximum, x[i]);
        }

        float sum = 0.0f;

        for (size_t i = 0; i < cols; i++) {

            y[i] = std::exp(x[i] - maximum);

            sum += y[i];
        }

        if (!(sum > 0.0f)) {
            throw std::runtime_error(
                "Softmax normalization failed"
            );
        }

        for (size_t i = 0; i < cols; i++) {
            y[i] /= sum;
        }
    }

    return output;
}

}
