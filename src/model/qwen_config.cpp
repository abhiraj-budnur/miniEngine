
#include "model/qwen_config.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>

using json = nlohmann::json;

namespace model {

QwenConfig QwenConfig::from_json(
    const std::string& path
) {

    std::ifstream file(path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "Cannot open model config: " + path
        );
    }

    json j;

    try {
        file >> j;
    }
    catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("Invalid config JSON: ") + e.what()
        );
    }

    QwenConfig config;

    config.model_type =
        j.value("model_type", std::string(""));

    config.architecture =
        j.value(
            "architectures",
            std::vector<std::string>{}
        ).empty()
        ? ""
        : j["architectures"][0].get<std::string>();

    config.vocab_size =
        j.value("vocab_size", size_t(0));

    config.hidden_size =
        j.value("hidden_size", size_t(0));

    config.intermediate_size =
        j.value("intermediate_size", size_t(0));

    config.num_hidden_layers =
        j.value("num_hidden_layers", size_t(0));

    config.num_attention_heads =
        j.value("num_attention_heads", size_t(0));

    config.num_key_value_heads =
        j.value(
            "num_key_value_heads",
            config.num_attention_heads
        );

    config.max_position_embeddings =
        j.value(
            "max_position_embeddings",
            size_t(0)
        );

    config.rms_norm_eps =
        j.value("rms_norm_eps", 1e-6f);

    config.rope_theta =
        j.value("rope_theta", 1000000.0f);

    config.bos_token_id =
        j.value("bos_token_id", -1);

    config.eos_token_id =
        j.value("eos_token_id", -1);

    config.tie_word_embeddings =
        j.value("tie_word_embeddings", false);

    if (config.num_attention_heads > 0) {

        if (config.hidden_size %
            config.num_attention_heads != 0) {

            throw std::runtime_error(
                "Hidden size is not divisible by attention heads"
            );
        }

        config.head_dim =
            config.hidden_size /
            config.num_attention_heads;
    }

    config.validate();

    return config;
}

void QwenConfig::validate() const {

    if (model_type != "qwen2") {
        throw std::runtime_error(
            "Unsupported model type: " + model_type
        );
    }

    if (hidden_size == 0 ||
        intermediate_size == 0 ||
        num_hidden_layers == 0 ||
        num_attention_heads == 0 ||
        num_key_value_heads == 0 ||
        vocab_size == 0) {

        throw std::runtime_error(
            "Missing required Qwen architecture parameters"
        );
    }

    if (hidden_size % num_attention_heads != 0) {
        throw std::runtime_error(
            "Invalid attention head configuration"
        );
    }

    if (num_attention_heads % num_key_value_heads != 0) {
        throw std::runtime_error(
            "Invalid grouped-query attention configuration"
        );
    }

    if (head_dim == 0 || head_dim % 2 != 0) {
        throw std::runtime_error(
            "Invalid RoPE head dimension"
        );
    }

    if (rms_norm_eps <= 0.0f ||
        rope_theta <= 0.0f) {

        throw std::runtime_error(
            "Invalid numerical model configuration"
        );
    }

    if (max_context_length == 0 ||
        max_position_embeddings == 0 ||
        max_context_length > max_position_embeddings) {

        throw std::runtime_error(
            "Invalid runtime context configuration"
        );
    }
}

}
