
cmake_minimum_required(VERSION 3.24)

project(MiniInference
    VERSION 0.2.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(nlohmann_json 3.10 REQUIRED)

# ==========================================
# MINI INFERENCE LIBRARY
# ==========================================

add_library(mini_inference

    # Core
    src/core/tensor.cpp
    src/core/tensor_io.cpp

    # Validation
    src/core/validation.cpp

    # Operations
    src/ops/matmul.cpp
    src/ops/rmsnorm.cpp
    src/ops/rope.cpp
    src/ops/softmax.cpp
    src/ops/swiglu.cpp

    # Model
    src/model/qwen_config.cpp
    src/model/safetensors_reader.cpp
    src/model/weight_mapper.cpp
    src/model/linear.cpp
    src/model/embedding.cpp
    src/model/attention.cpp
    src/model/transformer_block.cpp
    src/model/qwen_model.cpp

    # Memory
    src/memory/kv_cache.cpp
)

target_include_directories(
    mini_inference
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(
    mini_inference
    PUBLIC
    nlohmann_json::nlohmann_json
)

# ==========================================
# APPLE ACCELERATE
# ==========================================

if(APPLE)

    find_library(
        ACCELERATE_FRAMEWORK
        Accelerate
        REQUIRED
    )

    target_link_libraries(
        mini_inference
        PUBLIC
        ${ACCELERATE_FRAMEWORK}
    )

    message(STATUS "Apple Accelerate enabled")

endif()

# ==========================================
# COMPILER WARNINGS
# ==========================================

if(MSVC)

    target_compile_options(
        mini_inference PRIVATE /W4
    )

else()

    target_compile_options(
        mini_inference PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )

endif()

# ==========================================
# MAIN APPLICATION
# ==========================================

add_executable(
    mini_inference_app
    main.cpp
)

target_link_libraries(
    mini_inference_app
    PRIVATE mini_inference
)

# ==========================================
# NUMERICAL COMPARISON TOOL
# ==========================================

add_executable(
    compare
    tests/compare.cpp
)

target_link_libraries(
    compare
    PRIVATE mini_inference
)

# ==========================================
# EMBEDDING VALIDATION
# ==========================================

add_executable(
    test_embedding
    tests/test_embedding.cpp
)

target_link_libraries(
    test_embedding
    PRIVATE mini_inference
)

# ==========================================
# TESTING
# ==========================================

enable_testing()

add_test(
    NAME runtime_tests
    COMMAND mini_inference_app
)

# Embedding validation requires an external
# model checkpoint and reference files.
# Run test_embedding manually with its arguments.
