
#pragma once

#include <cstddef>
#include <mutex>
#include <unordered_map>

namespace memory {

struct GPUAllocation {
    void* pointer = nullptr;
    size_t bytes = 0;
    int device_id = 0;
};

struct GPUStats {
    size_t allocated_bytes = 0;
    size_t allocation_count = 0;
};

class GPUAllocator {

public:

    GPUAllocator() = default;

    ~GPUAllocator();

    GPUAllocation allocate(
        size_t bytes,
        int device_id = 0
    );

    void release(GPUAllocation& allocation);

    void copy_to_device(
        const GPUAllocation& destination,
        const void* source,
        size_t bytes
    );

    void copy_to_host(
        void* destination,
        const GPUAllocation& source,
        size_t bytes
    );

    void copy_device_to_device(
        const GPUAllocation& destination,
        const GPUAllocation& source,
        size_t bytes
    );

    GPUStats get_stats(int device_id = 0) const;

    static bool cuda_available();

private:

    mutable std::mutex mutex;

    std::unordered_map<int, GPUStats> statistics;
};

}
