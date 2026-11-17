
#pragma once

#include "core/tensor.h"

#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace model {

struct TensorInfo {

    std::string name;
    std::string dtype;

    std::vector<size_t> shape;

    uint64_t data_begin = 0;
    uint64_t data_end = 0;
};

class SafeTensorsReader {

public:

    explicit SafeTensorsReader(
        const std::string& filepath
    );

    bool contains(
        const std::string& name
    ) const;

    std::vector<std::string> tensor_names() const;

    const TensorInfo& get_info(
        const std::string& name
    ) const;

    core::Tensor load_tensor(
        const std::string& name
    );

private:

    std::string filepath_;

    std::ifstream file_;

    uint64_t data_start_ = 0;
    uint64_t file_size_ = 0;

    std::unordered_map<
        std::string,
        TensorInfo
    > tensors_;

    void read_header();

};

}
