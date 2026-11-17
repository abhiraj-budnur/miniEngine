
#include "model/attention.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <limits>
#include <utility>

namespace model {

Attention::Attention(
    core::Tensor q_weight,
    core::Tensor q_bias,
    core::Tensor k_weight,
    core::Tensor k_bias,
    core::Tensor v_weight,
    core::Tensor v_bias,
    core::Tensor o_weight,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim,
    size_t layer_index,
    float rope_theta
)
    : q_proj_(std::move(q_weight), std::move(q_bias)),
      k_proj_(std::move(k_weight), std::move(k_bias)),
      v_proj_(std::move(v_weight), std::move(v_bias)),
      o_proj_(std::move(o_weight)),
      num_heads_(num_heads),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim),
      hidden_size_(num_heads * head_dim),
      layer_index_(layer_index),
      rope_theta_(rope_theta) {

    if (num_heads == 0 ||
        num_kv_heads == 0 ||
        head_dim == 0 ||
        num_heads % num_kv_heads != 0) {

        throw std::invalid_argument(
            "Invalid GQA configuration"
        );
    }

    if (head_dim % 2 != 0) {
        throw std::invalid_argument(
            "RoPE requires an even head dimension"
        );
    }

    if (rope_theta <= 0.0f) {
        throw std::invalid_argument(
            "Invalid RoPE theta"
        );
    }
}

void Attention::apply_rope(
    float* vector,
    size_t position
) const {

    const size_t half = head_dim_ / 2;

    for (size_t i = 0; i < half; i++) {

        float exponent =
            static_cast<float>(2 * i) /
            static_cast<float>(head_dim_);

        float frequency =
            1.0f / std::pow(rope_theta_, exponent);

        float angle =
            static_cast<float>(position) * frequency;

        float cosine = std::cos(angle);
        float sine = std::sin(angle);

        float x1 = vector[i];
        float x2 = vector[i + half];

        vector[i] =
            x1 * cosine - x2 * sine;

        vector[i + half] =
            x2 * cosine + x1 * sine;
    }
}

core::Tensor Attention::forward(
    const core::Tensor& input,
    uint64_t sequence_id,
    size_t start_position,
    memory::KVCache& cache
) const {

    if (input.shape().size() != 2 ||
        input.shape()[1] != hidden_size_) {

        throw std::invalid_argument(
            "Invalid attention input shape"
        );
    }

    const size_t tokens = input.shape()[0];

    if (tokens == 0) {
        throw std::invalid_argument(
            "Empty attention input"
        );
    }

    const size_t kv_width =
        num_kv_heads_ * head_dim_;

    const size_t heads_per_kv =
        num_heads_ / num_kv_heads_;

    if (start_position + tokens >
        cache.max_sequence_length()) {

        throw std::out_of_range(
            "Attention exceeds KV cache context"
        );
    }

    if (cache.sequence_length(
        layer_index_,
        sequence_id
    ) != start_position) {

        throw std::runtime_error(
            "Attention position does not match KV cache"
        );
    }

    // Linear projections.

    core::Tensor q = q_proj_.forward(input);
    core::Tensor k = k_proj_.forward(input);
    core::Tensor v = v_proj_.forward(input);

    core::Tensor attention_output({
        tokens,
        hidden_size_
    });

    std::vector<float> current_key(kv_width);
    std::vector<float> current_value(kv_width);

    std::vector<float> scores(tokens == 0 ? 0 :
                              num_heads_ == 0 ? 0 : 1);

    const float scale =
        1.0f / std::sqrt(
            static_cast<float>(head_dim_)
        );

    for (size_t t = 0; t < tokens; t++) {

        const size_t position = start_position + t;

        // Apply RoPE to Q and K.
        // Q has 28 heads, K has 4 heads.

        for (size_t h = 0; h < num_heads_; h++) {

            float* q_head =
                q.data() +
                t * hidden_size_ +
                h * head_dim_;

            apply_rope(q_head, position);
        }

        for (size_t h = 0; h < num_kv_heads_; h++) {

            float* k_head =
                k.data() +
                t * kv_width +
                h * head_dim_;

            apply_rope(k_head, position);
        }

        // Append the current token's K/V to its
        // independent sequence cache.

        for (size_t i = 0; i < kv_width; i++) {

            current_key[i] =
                k.data()[t * kv_width + i];

            current_value[i] =
                v.data()[t * kv_width + i];
        }

        cache.append(
            layer_index_,
            sequence_id,
            position,
            current_key.data(),
            current_value.data()
        );

        const auto& sequence_cache =
            cache.get(layer_index_, sequence_id);

        const size_t context_length =
            sequence_cache.length;

        // Grouped Query Attention.

        for (size_t h = 0; h < num_heads_; h++) {

            const size_t kv_head =
                h / heads_per_kv;

            const float* q_head =
                q.data() +
                t * hidden_size_ +
                h * head_dim_;

            scores.resize(context_length);

            float max_score =
                -std::numeric_limits<float>::infinity();

            // QK^T.

            for (size_t p = 0;
                 p < context_length;
                 p++) {

                const float* key =
                    sequence_cache.keys.data() +
                    p * kv_width +
                    kv_head * head_dim_;

                float score = 0.0f;

                for (size_t d = 0;
                     d < head_dim_;
                     d++) {

                    score +=
                        q_head[d] * key[d];
                }

                score *= scale;

                scores[p] = score;

                max_score = std::max(
                    max_score,
                    score
                );
            }

            // Numerically stable softmax.

            float denominator = 0.0f;

            for (float& score : scores) {

                score = std::exp(
                    score - max_score
                );

                denominator += score;
            }

            if (denominator == 0.0f) {
                throw std::runtime_error(
                    "Invalid attention softmax"
                );
            }

            for (float& score : scores) {
                score /= denominator;
            }

            // Weighted sum of values.

            float* output_head =
                attention_output.data() +
                t * hidden_size_ +
                h * head_dim_;

            for (size_t d = 0;
                 d < head_dim_;
                 d++) {

                float result = 0.0f;

                for (size_t p = 0;
                     p < context_length;
                     p++) {

                    const float* value =
                        sequence_cache.values.data() +
                        p * kv_width +
                        kv_head * head_dim_;

                    result +=
                        scores[p] * value[d];
                }

                output_head[d] = result;
            }
        }
    }

    return o_proj_.forward(attention_output);
}

}
