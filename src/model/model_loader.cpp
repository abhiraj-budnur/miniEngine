
#include "model/model_loader.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <cstring>
#include <stdexcept>
#include <limits>

namespace model {

using json = nlohmann::json;

ModelLoader::ModelLoader(memory::MemoryManager& memory)
    : memory(memory) {
}

ModelConfig ModelLoader::load_config(
    const std::filesystem::path& directory
) {

    auto path = directory / "config.json";

    std::ifstream input(path);

    if (!input) {
        throw std::runtime_error("Cannot open config.json");
    }

    json j;

    input >> j;

    config.model_name = directory.filename().string();

    config.architecture = ModelArchitecture::QWEN2;

    config.vocab_size = j.at("vocab_size").get<int>();

    config.hidden_size = j.at("hidden_size").get<int>();

    config.intermediate_size =
        j.at("intermediate_size").get<int>();

    config.num_hidden_layers =
        j.at("num_hidden_layers").get<int>();

    config.num_attention_heads =
        j.at("num_attention_heads").get<int>();

    config.num_key_value_heads =
        j.value("num_key_value_heads",
                config.num_attention_heads);

    config.head_dim =
        j.value("head_dim",
                config.hidden_size /
                config.num_attention_heads);

    config.max_position_embeddings =
        j.at("max_position_embeddings").get<int>();

    config.rope_theta =
        j.value("rope_theta", 1000000.0f);

    config.rms_norm_eps =
        j.value("rms_norm_eps", 1e-6f);

    config.use_attention_bias =
        j.value("attention_bias", true);

    config.tie_word_embeddings =
        j.value("tie_word_embeddings", false);

    config.hidden_activation =
        j.value("hidden_act", std::string("silu"));

    if (!config.valid()) {
        throw std::runtime_error(
            "Invalid model configuration"
        );
    }

    return config;
}

std::vector<TensorMetadata> ModelLoader::read_metadata(
    const std::filesystem::path& file,
    uint64_t& data_start
) {

    std::ifstream input(file, std::ios::binary);

    if (!input) {
        throw std::runtime_error(
            "Cannot open Safetensors file"
        );
    }

    uint64_t header_size = 0;

    input.read(
        reinterpret_cast<char*>(&header_size),
        sizeof(header_size)
    );

    if (!input || header_size == 0 ||
        header_size > 100 * 1024 * 1024) {
        throw std::runtime_error(
            "Invalid Safetensors header size"
        );
    }

    auto file_size = std::filesystem::file_size(file);

    if (header_size > file_size - 8) {
        throw std::runtime_error(
            "Safetensors header exceeds file size"
        );
    }

    std::string header(header_size, '\0');

    input.read(header.data(), header_size);

    if (!input) {
        throw std::runtime_error(
            "Cannot read Safetensors header"
        );
    }

    data_start = 8 + header_size;

    json metadata = json::parse(header);

    std::vector<TensorMetadata> result;

    for (auto it = metadata.begin();
         it != metadata.end(); ++it) {

        if (it.key() == "__metadata__") {
            continue;
        }

        const auto& value = it.value();

        TensorMetadata tensor;

        tensor.name = it.key();

        tensor.dtype = value.at("dtype").get<std::string>();

        tensor.shape =
            value.at("shape").get<std::vector<int64_t>>();

        auto offsets =
            value.at("data_offsets").get<std::vector<uint64_t>>();

        if (offsets.size() != 2 ||
            offsets[0] > offsets[1] ||
            offsets[1] > file_size - data_start) {

            throw std::runtime_error(
                "Invalid Safetensors tensor offsets"
            );
        }

        tensor.data_begin = offsets[0];

        tensor.data_end = offsets[1];

        result.push_back(std::move(tensor));
    }

    return result;
}

void ModelLoader::load_safetensors_file(
    const std::filesystem::path& file
) {

    uint64_t data_start = 0;

    auto metadata = read_metadata(file, data_start);

    std::ifstream input(file, std::ios::binary);

    if (!input) {
        throw std::runtime_error(
            "Cannot open Safetensors data"
        );
    }

    const uint64_t file_size =
        std::filesystem::file_size(file);

    for (const auto& info : metadata) {

        core::DataType dtype;

        if (info.dtype == "F32") {
            dtype = core::DataType::FP32;
        }
        else if (info.dtype == "F16") {
            dtype = core::DataType::FP16;
        }
        else if (info.dtype == "BF16") {
            dtype = core::DataType::BF16;
        }
        else if (info.dtype == "I8") {
            dtype = core::DataType::INT8;
        }
        else if (info.dtype == "U8") {
            dtype = core::DataType::UINT8;
        }
        else if (info.dtype == "I32") {
            dtype = core::DataType::INT32;
        }
        else {
            throw std::runtime_error(
                "Unsupported Safetensors dtype: " + info.dtype
            );
        }

        size_t elements = 1;

        for (int64_t dim : info.shape) {

            if (dim < 0 ||
                (dim != 0 &&
                 elements >
                 std::numeric_limits<size_t>::max() /
                 static_cast<size_t>(dim))) {

                throw std::runtime_error(
                    "Invalid tensor shape"
                );
            }

            elements *= static_cast<size_t>(dim);
        }

        size_t bytes = info.data_end - info.data_begin;

        if (elements != 0 &&
            core::dtype_size(dtype) >
            std::numeric_limits<size_t>::max() / elements) {

            throw std::runtime_error(
                "Tensor size overflow"
            );
        }

        if (bytes != elements * core::dtype_size(dtype)) {
            throw std::runtime_error(
                "Tensor metadata and data size mismatch"
            );
        }

        auto tensor = memory.allocate_tensor(
            info.shape,
            dtype,
            core::Device::cpu()
        );

        input.seekg(
            static_cast<std::streamoff>(
                data_start + info.data_begin
            )
        );

        if (!input ||
            info.data_end > file_size - data_start) {

            throw std::runtime_error(
                "Invalid tensor file position"
            );
        }

        input.read(
            static_cast<char*>(tensor.data()),
            static_cast<std::streamsize>(bytes)
        );

        if (!input) {
            throw std::runtime_error(
                "Failed to read tensor: " + info.name
            );
        }

        if (weights.count(info.name)) {
            throw std::runtime_error(
                "Duplicate tensor name: " + info.name
            );
        }

        weights.emplace(info.name, std::move(tensor));
    }
}

void ModelLoader::load_weights(
    const std::filesystem::path& directory
) {

    auto index_path =
        directory / "model.safetensors.index.json";

    if (std::filesystem::exists(index_path)) {

        std::ifstream input(index_path);

        json index;

        input >> index;

        auto weight_map = index.at("weight_map");

        std::unordered_map<
            std::string,
            bool
        > loaded_files;

        for (auto it = weight_map.begin();
             it != weight_map.end(); ++it) {

            loaded_files[it.value().get<std::string>()] = true;
        }

        for (const auto& entry : loaded_files) {
            load_safetensors_file(directory / entry.first);
        }

    } else {

        for (const auto& entry :
             std::filesystem::directory_iterator(directory)) {

            if (entry.path().extension() == ".safetensors") {
                load_safetensors_file(entry.path());
            }
        }
    }

    if (weights.empty()) {
        throw std::runtime_error(
            "No model weights were loaded"
        );
    }
}

const core::Tensor& ModelLoader::get_weight(
    const std::string& name
) const {

    auto it = weights.find(name);

    if (it == weights.end()) {
        throw std::runtime_error(
            "Weight not found: " + name
        );
    }

    return it->second;
}

bool ModelLoader::has_weight(
    const std::string& name
) const {
    return weights.count(name) != 0;
}

size_t ModelLoader::weight_count() const {
    return weights.size();
}

const ModelConfig& ModelLoader::get_config() const {
    return config;
}

}
