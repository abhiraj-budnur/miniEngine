
#include "core/tensor_io.h"
#include "model/embedding.h"
#include "model/weight_mapper.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<int32_t> load_token_ids(
    const std::string& path
) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);

    if (!file) {
        throw std::runtime_error(
            "Cannot open input_ids.bin"
        );
    }

    auto bytes = file.tellg();

    if (bytes <= 0 ||
        bytes % sizeof(int32_t) != 0) {
        throw std::runtime_error(
            "Invalid token ID file"
        );
    }

    std::vector<int32_t> ids(
        static_cast<size_t>(bytes) / sizeof(int32_t)
    );

    file.seekg(0);

    file.read(
        reinterpret_cast<char*>(ids.data()),
        bytes
    );

    if (!file) {
        throw std::runtime_error(
            "Failed reading token IDs"
        );
    }

    return ids;
}

int main(int argc, char** argv) {

    if (argc != 3) {
        std::cerr
            << "Usage: test_embedding <model_dir> <reference_dir>\n";

        return 1;
    }

    try {

        const std::string model_dir = argv[1];
        const std::string reference_dir = argv[2];

        std::cout << "Loading token IDs...\n";

        auto token_ids = load_token_ids(
            reference_dir + "/input_ids.bin"
        );

        std::cout << "Token count: "
                  << token_ids.size() << '\n';

        std::cout << "Loading Qwen embedding weights...\n";

        model::WeightMapper mapper(model_dir);
        

        const std::string weight_name =
            "model.embed_tokens.weight";

        if (!mapper.contains(weight_name)) {
            throw std::runtime_error(
                "Embedding weights not found"
            );
        }

        core::Tensor embedding_weight =
            mapper.get(weight_name);

        if (embedding_weight.empty()) {
            throw std::runtime_error(
                "Embedding weight tensor is empty"
            );
        }

        std::cout << "Embedding weight shape: ";

        for (auto dimension : embedding_weight.shape()) {
            std::cout << dimension << " ";
        }

        std::cout << '\n';

        if (embedding_weight.dtype() != core::DType::FP32) {
            throw std::runtime_error(
                "Expected FP32 embedding weights"
            );
        }

        std::cout << "Running C++ embedding...\n";

        model::Embedding embedding(
            std::move(embedding_weight)
        );

        core::Tensor output = embedding.forward(token_ids);

        std::cout << "Output shape: ";

        for (auto dimension : output.shape()) {
            std::cout << dimension << " ";
        }

        std::cout << '\n';

        const std::string output_path =
            "tests/actual/embedding.bin";

        if (!core::save_tensor_binary(
                output,
                output_path
            )) {
            throw std::runtime_error(
                "Failed to export embedding output"
            );
        }

        std::cout << "Embedding export successful!\n";
        std::cout << "Saved: " << output_path << '\n';

        return 0;

    } catch (const std::exception& e) {

        std::cerr << "ERROR: " << e.what() << '\n';

        return 1;
    }
}
