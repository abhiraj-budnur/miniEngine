
#include "memory/gpu_allocator.h"

#include <stdexcept>

#ifndef MINI_INFERENCE_USE_CUDA

namespace memory {

GPUAllocator::~GPUAllocator() = default;

GPUAllocation GPUAllocator::allocate(
    size_t,
    int
) {
    throw std::runtime_error(
        "CUDA support is not enabled"
    );
}

void GPUAllocator::release(GPUAllocation&) {
}

void GPUAllocator::copy_to_device(
    const GPUAllocation&,
    const void*,
    size_t
) {
    throw std::runtime_error(
        "CUDA support is not enabled"
    );
}

void GPUAllocator::copy_to_host(
    void*,
    const GPUAllocation&,
    size_t
) {
    throw std::runtime_error(
        "CUDA support is not enabled"
    );
}

void GPUAllocator::copy_device_to_device(
    const GPUAllocation&,
    const GPUAllocation&,
    size_t
) {
    throw std::runtime_error(
        "CUDA support is not enabled"
    );
}

GPUStats GPUAllocator::get_stats(int) const {
    return {};
}

bool GPUAllocator::cuda_available() {
    return false;
}

}

#endif
