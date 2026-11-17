
#pragma once

#include "core/tensor.h"
#include "model/model_config.h"
#include "memory/memory_manager.h"

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>

namespace model {

struct TensorMetadata {

    std::string name;

    std::string dtype;

    std::vector<int64_t> shape;

    uint64_t data_begin = 0;

    uint64_t data_end = 0;
};

class ModelLoader {

public:

    explicit ModelLoader(memory::MemoryManager& memory);

    ModelConfig load_config(
        const std::filesystem::path& model_directory
    );

    void load_weights(
        const std::filesystem::path& model_directory
    );

    const core::Tensor& get_weight(
        const std::string& name
    ) const;

    bool has_weight(
        const std::string& name
    ) const;

    size_t weight_count() const;

    const ModelConfig& get_config() const;

private:

    memory::MemoryManager& memory;

    ModelConfig config;

    std::unordered_map<
        std::string,
        core::Tensor
    > weights;

    void load_safetensors_file(
        const std::filesystem::path& file
    );

    std::vector<TensorMetadata> read_metadata(
        const std::filesystem::path& file,
        uint64_t& data_start
    );
};

}
