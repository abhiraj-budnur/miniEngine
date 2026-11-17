
#include "model/linear.h"

#include "ops/matmul.h"

#include <stdexcept>

namespace model {

Linear::Linear(
    core::Tensor weight,
    core::Tensor bias
)
    : weight_(std::move(weight)),
      bias_(std::move(bias)) {

    if (weight_.shape().size() != 2) {
        throw std::invalid_argument(
            "Linear weight must be 2D"
        );
    }

    if (!bias_.empty() &&
        bias_.numel() != weight_.shape()[0]) {

        throw std::invalid_argument(
            "Linear bias dimension mismatch"
        );
    }
}

core::Tensor Linear::forward(
    const core::Tensor& input
) const {

    if (input.shape().empty()) {
        throw std::invalid_argument(
            "Invalid linear input"
        );
    }

    size_t input_features =
        input.shape().back();

    size_t output_features =
        weight_.shape()[0];

    size_t weight_input_features =
        weight_.shape()[1];

    if (input_features != weight_input_features) {
        throw std::invalid_argument(
            "Linear input feature mismatch"
        );
    }

    size_t rows =
        input.numel() / input_features;

    std::vector<size_t> output_shape =
        input.shape();

    output_shape.back() = output_features;

    core::Tensor flattened_input({
        rows,
        input_features
    });

    for (size_t i = 0; i < input.numel(); i++) {
        flattened_input.data()[i] = input.data()[i];
    }

    // Hugging Face linear weights are stored as
    // [out_features, in_features].
    //
    // matmul expects [in_features, out_features],
    // so we construct the transposed weight.

    core::Tensor transposed_weight({
        weight_input_features,
        output_features
    });

    for (size_t i = 0; i < output_features; i++) {

        for (size_t j = 0; j < weight_input_features; j++) {

            transposed_weight.data()[
                j * output_features + i
            ] = weight_.data()[
                i * weight_input_features + j
            ];
        }
    }

    core::Tensor output = ops::matmul(
        flattened_input,
        transposed_weight
    );

    if (!bias_.empty()) {

        for (size_t row = 0; row < rows; row++) {

            for (size_t col = 0;
                 col < output_features;
                 col++) {

                output.data()[
                    row * output_features + col
                ] += bias_.data()[col];
            }
        }
    }

    // Reshape is unnecessary because storage is
    // already contiguous and the element count matches.
    //
    // We return a tensor with the desired shape.

    core::Tensor result(output_shape);

    for (size_t i = 0; i < output.numel(); i++) {
        result.data()[i] = output.data()[i];
    }

    return result;
}

}
