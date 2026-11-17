
#include "attention/attention.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace attention {

Attention::Attention(
    const AttentionConfig& cfg
)
    : config(cfg),
      scale(1.0f / std::sqrt(
          static_cast<float>(cfg.head_dim)
      )) {

    if (config.num_attention_heads <= 0 ||
        config.num_key_value_heads <= 0 ||
        config.head_dim <= 0) {

        throw std::invalid_argument(
            "Invalid attention configuration"
        );
    }

    if (config.num_attention_heads %
        config.num_key_value_heads != 0) {

        throw std::invalid_argument(
            "Query heads must be divisible by KV heads"
        );
    }
}

void Attention::softmax(
    float* values,
    int size
) const {

    if (size <= 0) {
        throw std::invalid_argument(
            "Invalid softmax size"
        );
    }

    float maximum = values[0];

    for (int i = 1; i < size; i++) {
        maximum = std::max(maximum, values[i]);
    }

    float sum = 0.0f;

    for (int i = 0; i < size; i++) {

        values[i] = std::exp(values[i] - maximum);

        sum += values[i];
    }

    if (sum <= 0.0f || !std::isfinite(sum)) {
        throw std::runtime_error(
            "Invalid softmax normalization"
        );
    }

    float inverse_sum = 1.0f / sum;

    for (int i = 0; i < size; i++) {
        values[i] *= inverse_sum;
    }
}

void Attention::forward(
    int layer,
    const float* query,
    const float* key,
    const float* value,
    cache::KVCache& kv_cache,
    int position,
    float* output
) const {

    if (!query || !key || !value || !output) {
        throw std::invalid_argument(
            "Null attention input"
        );
    }

    if (kv_cache.num_kv_heads() !=
            config.num_key_value_heads ||
        kv_cache.head_dim() != config.head_dim) {

        throw std::runtime_error(
            "Attention and KV cache dimensions mismatch"
        );
    }

    if (position != kv_cache.current_length()) {
        throw std::runtime_error(
            "Attention position does not match KV cache length"
        );
    }

    // Append current K and V to the cache.
    kv_cache.append(
        layer,
        key,
        value,
        position
    );

    int sequence_length = position + 1;

    int heads_per_group =
        config.num_attention_heads /
        config.num_key_value_heads;

    const float* cached_keys =
        kv_cache.get_keys(layer);

    const float* cached_values =
        kv_cache.get_values(layer);

    std::vector<float> scores(sequence_length);

    for (int q_head = 0;
         q_head < config.num_attention_heads;
         q_head++) {

        // Seven query heads share one KV head
        // for the Qwen2.5-7B configuration.
        int kv_head = q_head / heads_per_group;

        const float* q =
            query + q_head * config.head_dim;

        for (int t = 0; t < sequence_length; t++) {

            const float* k =
                cached_keys +
                (
                    static_cast<size_t>(t) *
                    config.num_key_value_heads +
                    kv_head
                ) * config.head_dim;

            float dot = 0.0f;

            for (int d = 0; d < config.head_dim; d++) {
                dot += q[d] * k[d];
            }

            scores[t] = dot * scale;
        }

        softmax(
            scores.data(),
            sequence_length
        );

        float* out =
            output + q_head * config.head_dim;

        std::fill(
            out,
            out + config.head_dim,
            0.0f
        );

        for (int t = 0; t < sequence_length; t++) {

            const float* v =
                cached_values +
                (
                    static_cast<size_t>(t) *
                    config.num_key_value_heads +
                    kv_head
                ) * config.head_dim;

            float probability = scores[t];

            for (int d = 0; d < config.head_dim; d++) {

                out[d] += probability * v[d];
            }
        }
    }
}

}
