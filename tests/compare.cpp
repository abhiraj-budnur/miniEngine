
#include "core/validation.h"

#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>

std::vector<float> load_binary(const char* path) {

    std::ifstream file(path, std::ios::binary | std::ios::ate);

    if (!file) {
        throw std::runtime_error(
            std::string("Cannot open: ") + path
        );
    }

    auto size = file.tellg();

    if (size <= 0 || size % sizeof(float) != 0) {
        throw std::runtime_error("Invalid binary file size");
    }

    std::vector<float> data(
        static_cast<std::size_t>(size) / sizeof(float)
    );

    file.seekg(0);

    file.read(
        reinterpret_cast<char*>(data.data()),
        size
    );

    if (!file) {
        throw std::runtime_error("Failed reading binary file");
    }

    return data;
}

int main(int argc, char** argv) {

    if (argc != 3) {
        std::cerr
            << "Usage: compare reference.bin actual.bin\n";
        return 1;
    }

    try {

        auto reference = load_binary(argv[1]);
        auto actual = load_binary(argv[2]);

        if (reference.size() != actual.size()) {
            throw std::runtime_error(
                "Tensor element counts do not match"
            );
        }

        auto result = validation::compare(
            reference.data(),
            actual.data(),
            reference.size()
        );

        std::cout << "\nValidation results\n";
        std::cout << "------------------\n";

        std::cout << "Elements: "
                  << result.elements << '\n';

        std::cout << "Max absolute error: "
                  << result.max_absolute_error << '\n';

        std::cout << "MSE: "
                  << result.mean_squared_error << '\n';

        std::cout << "RMSE: "
                  << result.root_mean_squared_error << '\n';

        std::cout << "Max relative error: "
                  << result.max_relative_error << '\n';

        std::cout << "Status: "
                  << (result.passed ? "PASS" : "FAIL")
                  << '\n';

        return result.passed ? 0 : 1;

    } catch (const std::exception& e) {

        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
