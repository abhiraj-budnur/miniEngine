
#pragma once

#include "model/safetensors_reader.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace model {

class WeightMapper {

public:

    explicit WeightMapper(
        const std::string& model_directory
        
    );
    std::string get_dtype(
        const std::string& tensor_name
    );
    bool contains(
        const std::string& tensor_name
    ) const;

    core::Tensor get(
        const std::string& tensor_name
    );

    std::vector<std::string> tensor_names() const;

private:

    std::string model_directory_;

    std::unordered_map<
        std::string,
        std::string
    > tensor_to_shard_;

    std::unordered_map<
        std::string,
        std::unique_ptr<SafeTensorsReader>
    > readers_;

    void load_index();

    SafeTensorsReader& get_reader(
        const std::string& shard
    );

};

}
