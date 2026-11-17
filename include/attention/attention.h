
#pragma once

#include "cache/kv_cache.h"

#include <vector>

namespace attention {

struct AttentionConfig {

    int num_attention_heads = 28;

    int num_key_value_heads = 4;

    int head_dim = 128;
};

class Attention {

public:

    explicit Attention(const AttentionConfig& config);

    // Single-token GQA attention.
    //
    // query: [num_attention_heads, head_dim]
    // key:   [num_key_value_heads, head_dim]
    // value: [num_key_value_heads, head_dim]
    //
    // output: [num_attention_heads, head_dim]

    void forward(
        int layer,
        const float* query,
        const float* key,
        const float* value,
        cache::KVCache& kv_cache,
        int position,
        float* output
    ) const;

private:

    AttentionConfig config;

    float scale;

    void softmax(
        float* values,
        int size
    ) const;
};

}
