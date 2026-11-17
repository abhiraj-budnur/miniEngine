
#include "model/model_config.h"

namespace model {

ModelConfig qwen25_7b_config() {

    ModelConfig config;

    config.architecture = ModelArchitecture::QWEN2;

    config.model_name = "Qwen2.5-7B-Instruct";

    config.vocab_size = 152064;

    config.hidden_size = 3584;

    config.intermediate_size = 18944;

    config.num_hidden_layers = 28;

    config.num_attention_heads = 28;

    config.num_key_value_heads = 4;

    config.head_dim = 128;

    config.max_position_embeddings = 32768;

    config.rope_theta = 1000000.0f;

    config.rms_norm_eps = 1e-6f;

    config.use_attention_bias = true;

    config.tie_word_embeddings = false;

    config.hidden_activation = "silu";

    config.default_max_new_tokens = 128;

    return config;
}

}
