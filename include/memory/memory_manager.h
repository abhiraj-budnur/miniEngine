
#pragma once

#include "core/device.h"
#include "core/dtype.h"
#include "core/tensor.h"

#include <cstddef>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace memory {

struct MemoryStats {

    size_t total_allocated = 0;

    size_t allocation_count = 0;
};

class MemoryManager {

public:

    MemoryManager() = default;

    ~MemoryManager() = default;

    // Allocate raw memory on a device.
    std::shared_ptr<core::TensorStorage> allocate(
        size_t bytes,
        core::Device device
    );

    // Allocate a tensor.
    core::Tensor allocate_tensor(
        const std::vector<int64_t>& shape,
        core::DataType dtype,
        core::Device device
    );

    // Memory statistics.
    MemoryStats get_stats(
        core::Device device
    ) const;

    // Explicitly release an allocation.
    void release(
        std::shared_ptr<core::TensorStorage>& storage
    );

private:

    struct DeviceKey {

        core::DeviceType type;

        int id;

        bool operator==(const DeviceKey& other) const {
            return type == other.type && id == other.id;
        }
    };

    struct DeviceKeyHash {
        size_t operator()(const DeviceKey& key) const {
            return (static_cast<size_t>(key.type) << 32)
                   ^ static_cast<size_t>(key.id);
        }
    };

    mutable std::mutex mutex;

    std::unordered_map<DeviceKey, MemoryStats, DeviceKeyHash>
        statistics;

};

}
