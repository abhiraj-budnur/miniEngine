
#include "model/qwen_model.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace model {

QwenModel::QwenModel(
    WeightMapper& mapper,
    size_t vocab_size,
    size_t hidden_size,
    size_t intermediate_size,
    size_t num_layers,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim,
    float rms_norm_eps,
    float rope_theta
)
    : embedding_(
          mapper.get("model.embed_tokens.weight")
      ),
      final_norm_weight_(
          mapper.get("model.norm.weight")
      ),
      lm_head_(
          mapper.get("lm_head.weight")
      ),
      vocab_size_(vocab_size),
      hidden_size_(hidden_size),
      rms_norm_eps_(rms_norm_eps) {

    if (num_layers == 0 ||
        num_heads * head_dim != hidden_size ||
        num_heads % num_kv_heads != 0) {

        throw std::invalid_argument(
            "Invalid Qwen model configuration"
        );
    }

    if (final_norm_weight_.numel() != hidden_size) {
        throw std::runtime_error(
            "Final RMSNorm weight mismatch"
        );
    }

    layers_.reserve(num_layers);

    for (size_t i = 0; i < num_layers; i++) {

        const std::string prefix =
            "model.layers." + std::to_string(i) + ".";

        auto block = std::make_unique<TransformerBlock>(

            i,

            // RMSNorm weights.
            mapper.get(
                prefix + "input_layernorm.weight"
            ),

            mapper.get(
                prefix + "post_attention_layernorm.weight"
            ),

            // Attention projections.
            mapper.get(
                prefix + "self_attn.q_proj.weight"
            ),

            mapper.get(
                prefix + "self_attn.q_proj.bias"
            ),

            mapper.get(
                prefix + "self_attn.k_proj.weight"
            ),

            mapper.get(
                prefix + "self_attn.k_proj.bias"
            ),

            mapper.get(
                prefix + "self_attn.v_proj.weight"
            ),

            mapper.get(
                prefix + "self_attn.v_proj.bias"
            ),

            mapper.get(
                prefix + "self_attn.o_proj.weight"
            ),

            // MLP projections.
            mapper.get(
                prefix + "mlp.gate_proj.weight"
            ),

            mapper.get(
                prefix + "mlp.up_proj.weight"
            ),

            mapper.get(
                prefix + "mlp.down_proj.weight"
            ),

            hidden_size,
            intermediate_size,
            num_heads,
            num_kv_heads,
            head_dim,
            rms_norm_eps,
            rope_theta
        );

        layers_.push_back(
            std::move(block)
        );
    }
}

core::Tensor QwenModel::rms_norm(
    const core::Tensor& input
) const {

    if (input.shape().empty() ||
        input.shape().back() != hidden_size_) {

        throw std::invalid_argument(
            "Final RMSNorm input mismatch"
        );
    }

    core::Tensor output(input.shape());

    size_t rows =
        input.numel() / hidden_size_;

    for (size_t row = 0; row < rows; row++) {

        const float* source =
            input.data() + row * hidden_size_;

        float* destination =
            output.data() + row * hidden_size_;

        float sum = 0.0f;

        for (size_t i = 0; i < hidden_size_; i++) {

            sum += source[i] * source[i];
        }

        float inverse_rms =
            1.0f / std::sqrt(
                sum / static_cast<float>(hidden_size_) +
                rms_norm_eps_
            );

        for (size_t i = 0; i < hidden_size_; i++) {

            destination[i] =
                source[i] *
                inverse_rms *
                final_norm_weight_.data()[i];
        }
    }

    return output;
}

core::Tensor QwenModel::forward(
    const std::vector<int32_t>& token_ids,
    uint64_t sequence_id,
    size_t start_position,
    memory::KVCache& cache
) const {

    if (token_ids.empty()) {
        throw std::invalid_argument(
            "Cannot run model on empty input"
        );
    }

    if (start_position + token_ids.size() >
        cache.max_sequence_length()) {

        throw std::out_of_range(
            "Model input exceeds context length"
        );
    }

    if (cache.num_layers() != layers_.size()) {
        throw std::runtime_error(
            "Model and KV cache layer counts differ"
        );
    }

    // Token IDs -> embeddings.

    core::Tensor hidden =
        embedding_.forward(token_ids);

    // Transformer stack.

    for (const auto& layer : layers_) {

        hidden = layer->forward(
            hidden,
            sequence_id,
            start_position,
            cache
        );
    }

    // Final normalization.

    hidden = rms_norm(hidden);

    // Hidden states -> vocabulary logits.

    core::Tensor logits =
        lm_head_.forward(hidden);

    return logits;
}

size_t QwenModel::vocab_size() const {
    return vocab_size_;
}

size_t QwenModel::num_layers() const {
    return layers_.size();
}

size_t QwenModel::hidden_size() const {
    return hidden_size_;
}

}
