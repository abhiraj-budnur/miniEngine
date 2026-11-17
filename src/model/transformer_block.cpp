
#include "model/transformer_block.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace model {

TransformerBlock::TransformerBlock(
    size_t layer_index,
    core::Tensor input_norm_weight,
    core::Tensor post_attention_norm_weight,
    core::Tensor q_weight,
    core::Tensor q_bias,
    core::Tensor k_weight,
    core::Tensor k_bias,
    core::Tensor v_weight,
    core::Tensor v_bias,
    core::Tensor o_weight,
    core::Tensor gate_weight,
    core::Tensor up_weight,
    core::Tensor down_weight,
    size_t hidden_size,
    size_t intermediate_size,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim,
    float rms_norm_eps,
    float rope_theta
)
    : input_norm_weight_(
          std::move(input_norm_weight)
      ),
      post_attention_norm_weight_(
          std::move(post_attention_norm_weight)
      ),
      attention_(
          std::move(q_weight),
          std::move(q_bias),
          std::move(k_weight),
          std::move(k_bias),
          std::move(v_weight),
          std::move(v_bias),
          std::move(o_weight),
          num_heads,
          num_kv_heads,
          head_dim,
          layer_index,
          rope_theta
      ),
      gate_proj_(std::move(gate_weight)),
      up_proj_(std::move(up_weight)),
      down_proj_(std::move(down_weight)),
      hidden_size_(hidden_size),
      intermediate_size_(intermediate_size),
      rms_norm_eps_(rms_norm_eps) {

    if (hidden_size == 0 ||
        intermediate_size == 0 ||
        rms_norm_eps <= 0.0f) {

        throw std::invalid_argument(
            "Invalid transformer configuration"
        );
    }
}

core::Tensor TransformerBlock::rms_norm(
    const core::Tensor& input,
    const core::Tensor& weight
) const {

    if (input.shape().empty() ||
        input.shape().back() != hidden_size_ ||
        weight.numel() != hidden_size_) {

        throw std::invalid_argument(
            "RMSNorm dimension mismatch"
        );
    }

    core::Tensor output(input.shape());

    const size_t rows =
        input.numel() / hidden_size_;

    for (size_t row = 0; row < rows; row++) {

        const float* source =
            input.data() + row * hidden_size_;

        float* destination =
            output.data() + row * hidden_size_;

        float sum = 0.0f;

        for (size_t i = 0;
             i < hidden_size_;
             i++) {

            sum += source[i] * source[i];
        }

        float inverse_rms =
            1.0f / std::sqrt(
                sum / static_cast<float>(hidden_size_) +
                rms_norm_eps_
            );

        for (size_t i = 0;
             i < hidden_size_;
             i++) {

            destination[i] =
                source[i] *
                inverse_rms *
                weight.data()[i];
        }
    }

    return output;
}

core::Tensor TransformerBlock::forward(
    const core::Tensor& input,
    uint64_t sequence_id,
    size_t start_position,
    memory::KVCache& cache
) const {

    if (input.shape().size() != 2 ||
        input.shape()[1] != hidden_size_) {

        throw std::invalid_argument(
            "Invalid transformer input"
        );
    }

    // Pre-attention RMSNorm.

    core::Tensor normalized =
        rms_norm(input, input_norm_weight_);

    // Self-attention.

    core::Tensor attention_output =
        attention_.forward(
            normalized,
            sequence_id,
            start_position,
            cache
        );

    // First residual connection.

    core::Tensor hidden(input.shape());

    for (size_t i = 0; i < input.numel(); i++) {

        hidden.data()[i] =
            input.data()[i] +
            attention_output.data()[i];
    }

    // Post-attention RMSNorm.

    normalized = rms_norm(
        hidden,
        post_attention_norm_weight_
    );

    // SwiGLU MLP.

    core::Tensor gate =
        gate_proj_.forward(normalized);

    core::Tensor up =
        up_proj_.forward(normalized);

    if (gate.shape().back() != intermediate_size_ ||
        up.shape() != gate.shape()) {

        throw std::runtime_error(
            "MLP projection dimension mismatch"
        );
    }

    // SiLU(gate) * up.

    for (size_t i = 0; i < gate.numel(); i++) {

        float x = gate.data()[i];

        float silu =
            x / (1.0f + std::exp(-x));

        gate.data()[i] = silu * up.data()[i];
    }

    core::Tensor mlp_output =
        down_proj_.forward(gate);

    // Second residual connection.

    core::Tensor output(input.shape());

    for (size_t i = 0; i < input.numel(); i++) {

        output.data()[i] =
            hidden.data()[i] +
            mlp_output.data()[i];
    }

    return output;
}

}
