
#include "core/tensor.h"

#include "ops/matmul.h"
#include "ops/rmsnorm.h"
#include "ops/rope.h"
#include "ops/softmax.h"
#include "ops/swiglu.h"

#include "model/qwen_config.h"
#include "model/weight_mapper.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cstddef>

using namespace core;
using namespace ops;

void test_matmul()
{

    Tensor A({2, 2});
    Tensor B({2, 2});

    float a[] = {1, 2, 3, 4};
    float b[] = {5, 6, 7, 8};

    for (int i = 0; i < 4; i++)
    {
        A.data()[i] = a[i];
        B.data()[i] = b[i];
    }

    Tensor C = matmul(A, B);

    const float expected[] = {
        19, 22, 43, 50};

    for (int i = 0; i < 4; i++)
    {

        if (std::abs(
                C.data()[i] - expected[i]) > 1e-4f)
        {

            throw std::runtime_error("Matmul failed");
        }
    }

    std::cout << "[PASS] Matmul\n";
}

void test_rmsnorm()
{

    Tensor input({1, 4});
    Tensor weight({4});

    float values[] = {1, 2, 3, 4};

    for (int i = 0; i < 4; i++)
    {
        input.data()[i] = values[i];
        weight.data()[i] = 1.0f;
    }

    Tensor output = rmsnorm(input, weight);

    float rms = std::sqrt(30.0f / 4.0f + 1e-6f);

    for (int i = 0; i < 4; i++)
    {

        if (std::abs(
                output.data()[i] - values[i] / rms) > 1e-4f)
        {

            throw std::runtime_error("RMSNorm failed");
        }
    }

    std::cout << "[PASS] RMSNorm\n";
}

void test_softmax()
{

    Tensor input({1, 3});

    input.data()[0] = 1;
    input.data()[1] = 2;
    input.data()[2] = 3;

    Tensor output = softmax(input);

    float sum = 0;

    for (int i = 0; i < 3; i++)
    {
        sum += output.data()[i];
    }

    if (std::abs(sum - 1.0f) > 1e-5f)
    {
        throw std::runtime_error("Softmax failed");
    }

    std::cout << "[PASS] Softmax\n";
}

void test_rope()
{

    Tensor input({1, 4});

    float values[] = {1, 2, 3, 4};

    for (int i = 0; i < 4; i++)
    {
        input.data()[i] = values[i];
    }

    apply_rope(input, 0);

    for (int i = 0; i < 4; i++)
    {

        if (std::abs(
                input.data()[i] - values[i]) > 1e-5f)
        {

            throw std::runtime_error("RoPE failed");
        }
    }

    std::cout << "[PASS] RoPE\n";
}

void test_swiglu()
{

    Tensor gate({1, 3});
    Tensor up({1, 3});

    for (int i = 0; i < 3; i++)
    {
        gate.data()[i] = 1.0f;
        up.data()[i] = 2.0f;
    }

    Tensor output = swiglu(gate, up);

    float expected =
        2.0f / (1.0f + std::exp(-1.0f));

    for (int i = 0; i < 3; i++)
    {

        if (std::abs(
                output.data()[i] - expected) > 1e-5f)
        {

            throw std::runtime_error("SwiGLU failed");
        }
    }

    std::cout << "[PASS] SwiGLU\n";
}

void run_tests()
{

    std::cout << "\nRunning numerical tests...\n\n";

    test_matmul();
    test_rmsnorm();
    test_softmax();
    test_rope();
    test_swiglu();

    std::cout << "\nAll numerical tests passed.\n";
}

void inspect_model(const std::string &directory)
{

    std::cout << "\nLoading Qwen configuration...\n";

    auto config = model::QwenConfig::from_json(
        directory + "/config.json");

    std::cout << "\nModel configuration\n";
    std::cout << "-------------------\n";

    std::cout << "Model type: "
              << config.model_type << '\n';

    std::cout << "Architecture: "
              << config.architecture << '\n';

    std::cout << "Vocabulary: "
              << config.vocab_size << '\n';

    std::cout << "Hidden size: "
              << config.hidden_size << '\n';

    std::cout << "Intermediate size: "
              << config.intermediate_size << '\n';

    std::cout << "Layers: "
              << config.num_hidden_layers << '\n';

    std::cout << "Attention heads: "
              << config.num_attention_heads << '\n';

    std::cout << "KV heads: "
              << config.num_key_value_heads << '\n';

    std::cout << "Head dimension: "
              << config.head_dim << '\n';

    std::cout << "Runtime context: "
              << config.max_context_length << '\n';

    std::cout << "\nInspecting checkpoint index...\n";

    model::WeightMapper mapper(directory);

    std::cout
        << "Embedding checkpoint dtype: "
        << mapper.get_dtype("model.embed_tokens.weight")
        << '\n';

    // DEBUG: Inspect embedding checkpoint dtype
    std::cout
        << "Embedding checkpoint dtype: "
        << mapper.get_dtype("model.embed_tokens.weight")
        << '\n';

    auto names = mapper.tensor_names();

    std::cout << "Indexed tensors: "
              << names.size() << '\n';

    for (size_t i = 0;
         i < std::min(names.size(), size_t(15));
         i++)
    {

        std::cout << "  " << names[i] << '\n';
    }

    std::cout << "\nCheckpoint inspection completed.\n";
}

int main(int argc, char *argv[])
{

    try
    {

        std::cout << "\n";
        std::cout << "=================================\n";
        std::cout << "       MINI INFERENCE ENGINE     \n";
        std::cout << "=================================\n";

        if (argc >= 3 &&
            std::string(argv[1]) == "--inspect")
        {

            inspect_model(argv[2]);
        }
        else
        {

            std::cout << "Backend: CPU FP32 reference\n";

            run_tests();
        }

        return 0;
    }
    catch (const std::exception &e)
    {

        std::cerr << "\nERROR: "
                  << e.what() << '\n';

        return 1;
    }
}
