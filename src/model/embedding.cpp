#include "model/embedding.h"

#include <stdexcept>
#include <utility>

namespace model {

Embedding::Embedding(core::Tensor weight)
    : weight_(std::move(weight)) {

    if (weight_.shape().size() != 2) {
        throw std::invalid_argument(
            "Embedding weight must be 2D"
        );
    }

    if (weight_.dtype() != core::DType::FP32) {
        throw std::invalid_argument(
            "Embedding currently requires FP32 weights"
        );
    }
}

core::Tensor Embedding::forward(
    const std::vector<int32_t>& token_ids
) const {

    size_t vocab_size = weight_.shape()[0];
    size_t hidden_size = weight_.shape()[1];

    core::Tensor output({
        token_ids.size(),
        hidden_size
    });

    for (size_t i = 0; i < token_ids.size(); i++) {

        int32_t token = token_ids[i];

        if (token < 0 ||
            static_cast<size_t>(token) >= vocab_size) {

            throw std::out_of_range(
                "Token ID outside embedding vocabulary"
            );
        }

        const float* source =
            weight_.data() +
            static_cast<size_t>(token) * hidden_size;

        float* destination =
            output.data() + i * hidden_size;

        for (size_t j = 0; j < hidden_size; j++) {
            destination[j] = source[j];
        }
    }

    return output;
}

}