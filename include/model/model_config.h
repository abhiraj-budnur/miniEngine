
#pragma once

#include <string>
#include <cstdint>

namespace model {

enum class ModelArchitecture {
    QWEN2,
    UNKNOWN
};

struct ModelConfig {

    // Architecture
    ModelArchitecture architecture =
        ModelArchitecture::UNKNOWN;

    std::string model_name;

    // Transformer dimensions
    int vocab_size = 0;

    int hidden_size = 0;

    int intermediate_size = 0;

    int num_hidden_layers = 0;

    int num_attention_heads = 0;

    int num_key_value_heads = 0;

    int head_dim = 0;

    // Position embeddings
    int max_position_embeddings = 0;

    float rope_theta = 1000000.0f;

    float rms_norm_eps = 1e-6f;

    // Model behavior
    bool use_attention_bias = true;

    bool tie_word_embeddings = false;

    std::string hidden_activation = "silu";

    // Default inference settings
    int default_max_new_tokens = 128;

    bool valid() const {
        return vocab_size > 0 &&
               hidden_size > 0 &&
               intermediate_size > 0 &&
               num_hidden_layers > 0 &&
               num_attention_heads > 0 &&
               num_key_value_heads > 0 &&
               head_dim > 0;
    }
};

// Returns the initial Qwen2.5-7B configuration.
ModelConfig qwen25_7b_config();

}
