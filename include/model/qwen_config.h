
#pragma once

#include <cstddef>
#include <string>

namespace model {

struct QwenConfig {

    std::string model_type;
    std::string architecture;

    size_t vocab_size = 0;
    size_t hidden_size = 0;
    size_t intermediate_size = 0;

    size_t num_hidden_layers = 0;
    size_t num_attention_heads = 0;
    size_t num_key_value_heads = 0;

    size_t head_dim = 0;

    size_t max_position_embeddings = 0;
    size_t max_context_length = 4096;

    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;

    int bos_token_id = -1;
    int eos_token_id = -1;

    bool tie_word_embeddings = false;

    static QwenConfig from_json(
        const std::string& path
    );

    void validate() const;

};

}
