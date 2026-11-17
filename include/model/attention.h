
#pragma once

#include "core/tensor.h"
#include "model/linear.h"
#include "memory/kv_cache.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace model {

class Attention {

public:

    Attention(
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
        float rope_theta = 1000000.0f
    );

    core::Tensor forward(
        const core::Tensor& input,
        uint64_t sequence_id,
        size_t start_position,
        memory::KVCache& cache
    ) const;

private:

    Linear q_proj_;
    Linear k_proj_;
    Linear v_proj_;
    Linear o_proj_;

    size_t num_heads_;
    size_t num_kv_heads_;
    size_t head_dim_;
    size_t hidden_size_;
    size_t layer_index_;

    float rope_theta_;

    void apply_rope(
        float* vector,
        size_t position
    ) const;

};

}
