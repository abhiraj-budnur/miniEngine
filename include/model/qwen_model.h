
#pragma once

#include "core/tensor.h"
#include "model/embedding.h"
#include "model/transformer_block.h"
#include "model/weight_mapper.h"
#include "memory/kv_cache.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace model {

class QwenModel {

public:

    QwenModel(
        WeightMapper& mapper,
        size_t vocab_size,
        size_t hidden_size,
        size_t intermediate_size,
        size_t num_layers,
        size_t num_heads,
        size_t num_kv_heads,
        size_t head_dim,
        float rms_norm_eps = 1e-6f,
        float rope_theta = 1000000.0f
    );

    core::Tensor forward(
        const std::vector<int32_t>& token_ids,
        uint64_t sequence_id,
        size_t start_position,
        memory::KVCache& cache
    ) const;

    size_t vocab_size() const;

    size_t num_layers() const;

    size_t hidden_size() const;

private:

    Embedding embedding_;

    std::vector<
        std::unique_ptr<TransformerBlock>
    > layers_;

    core::Tensor final_norm_weight_;

    Linear lm_head_;

    size_t vocab_size_;
    size_t hidden_size_;

    float rms_norm_eps_;

    core::Tensor rms_norm(
        const core::Tensor& input
    ) const;

};

}
