
#pragma once

#include "core/tensor.h"
#include "model/attention.h"
#include "memory/kv_cache.h"

#include <cstddef>
#include <cstdint>

namespace model {

class TransformerBlock {

public:

    TransformerBlock(
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
        float rms_norm_eps = 1e-6f,
        float rope_theta = 1000000.0f
    );

    core::Tensor forward(
        const core::Tensor& input,
        uint64_t sequence_id,
        size_t start_position,
        memory::KVCache& cache
    ) const;

private:

    core::Tensor input_norm_weight_;
    core::Tensor post_attention_norm_weight_;

    Attention attention_;

    Linear gate_proj_;
    Linear up_proj_;
    Linear down_proj_;

    size_t hidden_size_;
    size_t intermediate_size_;

    float rms_norm_eps_;

    core::Tensor rms_norm(
        const core::Tensor& input,
        const core::Tensor& weight
    ) const;

};

}
