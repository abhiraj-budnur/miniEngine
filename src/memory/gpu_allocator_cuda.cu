
#include "memory/gpu_allocator.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace memory {

static void check_cuda(
    cudaError_t status,
    const char* operation
) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " +
            cudaGetErrorString(status)
        );
    }
}

GPUAllocator::~GPUAllocator() = default;

GPUAllocation GPUAllocator::allocate(
    size_t bytes,
    int device_id
) {

    if (bytes == 0) {
        throw std::invalid_argument(
            "Cannot allocate zero bytes"
        );
    }

    check_cuda(
        cudaSetDevice(device_id),
        "cudaSetDevice"
    );

    void* pointer = nullptr;

    check_cuda(
        cudaMalloc(&pointer, bytes),
        "cudaMalloc"
    );

    GPUAllocation allocation;

    allocation.pointer = pointer;
    allocation.bytes = bytes;
    allocation.device_id = device_id;

    {
        std::lock_guard<std::mutex> lock(mutex);

        auto& stats = statistics[device_id];

        stats.allocated_bytes += bytes;
        stats.allocation_count++;
    }

    return allocation;
}

void GPUAllocator::release(
    GPUAllocation& allocation
) {

    if (!allocation.pointer) {
        return;
    }

    check_cuda(
        cudaSetDevice(allocation.device_id),
        "cudaSetDevice"
    );

    check_cuda(
        cudaFree(allocation.pointer),
        "cudaFree"
    );

    {
        std::lock_guard<std::mutex> lock(mutex);

        auto& stats = statistics[allocation.device_id];

        if (stats.allocated_bytes >= allocation.bytes) {
            stats.allocated_bytes -= allocation.bytes;
        }

        if (stats.allocation_count > 0) {
            stats.allocation_count--;
        }
    }

    allocation.pointer = nullptr;
    allocation.bytes = 0;
}

void GPUAllocator::copy_to_device(
    const GPUAllocation& destination,
    const void* source,
    size_t bytes
) {

    if (bytes > destination.bytes) {
        throw std::runtime_error(
            "Copy exceeds GPU allocation"
        );
    }

    check_cuda(
        cudaSetDevice(destination.device_id),
        "cudaSetDevice"
    );

    check_cuda(
        cudaMemcpy(
            destination.pointer,
            source,
            bytes,
            cudaMemcpyHostToDevice
        ),
        "Host to device copy"
    );
}

void GPUAllocator::copy_to_host(
    void* destination,
    const GPUAllocation& source,
    size_t bytes
) {

    if (bytes > source.bytes) {
        throw std::runtime_error(
            "Copy exceeds GPU allocation"
        );
    }

    check_cuda(
        cudaSetDevice(source.device_id),
        "cudaSetDevice"
    );

    check_cuda(
        cudaMemcpy(
            destination,
            source.pointer,
            bytes,
            cudaMemcpyDeviceToHost
        ),
        "Device to host copy"
    );
}

void GPUAllocator::copy_device_to_device(
    const GPUAllocation& destination,
    const GPUAllocation& source,
    size_t bytes
) {

    if (bytes > destination.bytes ||
        bytes > source.bytes) {

        throw std::runtime_error(
            "Device copy exceeds allocation"
        );
    }

    if (destination.device_id != source.device_id) {
        throw std::runtime_error(
            "Cross-device copy is not supported here"
        );
    }

    check_cuda(
        cudaSetDevice(destination.device_id),
        "cudaSetDevice"
    );

    check_cuda(
        cudaMemcpy(
            destination.pointer,
            source.pointer,
            bytes,
            cudaMemcpyDeviceToDevice
        ),
        "Device to device copy"
    );
}

GPUStats GPUAllocator::get_stats(int device_id) const {

    std::lock_guard<std::mutex> lock(mutex);

    auto it = statistics.find(device_id);

    if (it == statistics.end()) {
        return {};
    }

    return it->second;
}

bool GPUAllocator::cuda_available() {

    int count = 0;

    return cudaGetDeviceCount(&count) == cudaSuccess &&
           count > 0;
}

}
