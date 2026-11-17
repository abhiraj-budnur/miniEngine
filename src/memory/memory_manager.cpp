
#include "memory/memory_manager.h"

#include <new>
#include <stdexcept>
#include <limits>

namespace memory {

std::shared_ptr<core::TensorStorage>
MemoryManager::allocate(
    size_t bytes,
    core::Device device
) {

    if (bytes == 0) {
        throw std::invalid_argument(
            "Cannot allocate zero bytes"
        );
    }

    if (device.type != core::DeviceType::CPU) {
        throw std::runtime_error(
            "CUDA and Metal allocators are not implemented yet"
        );
    }

    auto* ptr = new std::byte[bytes];

    auto storage = std::make_shared<core::TensorStorage>();

    storage->data = ptr;

    storage->bytes = bytes;

    storage->device = device;

    // The allocation is automatically released
    // when the final shared owner disappears.
    storage->owner = std::shared_ptr<void>(
        ptr,
        [](void* p) {
            delete[] static_cast<std::byte*>(p);
        }
    );

    {
        std::lock_guard<std::mutex> lock(mutex);

        DeviceKey key{device.type, device.device_id};

        auto& stats = statistics[key];

        if (stats.total_allocated >
            std::numeric_limits<size_t>::max() - bytes) {
            throw std::overflow_error("Memory statistics overflow");
        }

        stats.total_allocated += bytes;

        stats.allocation_count++;
    }

    return storage;
}

core::Tensor MemoryManager::allocate_tensor(
    const std::vector<int64_t>& shape,
    core::DataType dtype,
    core::Device device
) {

    if (shape.empty()) {
        throw std::invalid_argument(
            "Tensor shape cannot be empty"
        );
    }

    size_t elements = 1;

    for (int64_t dim : shape) {

        if (dim < 0) {
            throw std::invalid_argument(
                "Negative tensor dimension"
            );
        }

        size_t d = static_cast<size_t>(dim);

        if (d != 0 &&
            elements > std::numeric_limits<size_t>::max() / d) {
            throw std::overflow_error("Tensor size overflow");
        }

        elements *= d;
    }

    size_t bytes;

    if (dtype == core::DataType::INT4) {
        bytes = elements / 2 + elements % 2;
    } else {
        size_t element_size = core::dtype_size(dtype);

        if (elements != 0 &&
            element_size >
                std::numeric_limits<size_t>::max() / elements) {
            throw std::overflow_error("Tensor byte size overflow");
        }

        bytes = elements * element_size;
    }

    if (bytes == 0) {
        throw std::invalid_argument(
            "Cannot allocate an empty tensor"
        );
    }

    auto storage = allocate(bytes, device);

    return core::Tensor(
        storage,
        shape,
        dtype
    );
}

MemoryStats MemoryManager::get_stats(
    core::Device device
) const {

    std::lock_guard<std::mutex> lock(mutex);

    DeviceKey key{device.type, device.device_id};

    auto it = statistics.find(key);

    if (it == statistics.end()) {
        return {};
    }

    // These are cumulative allocation statistics
    // in this initial implementation.
    return it->second;
}

void MemoryManager::release(
    std::shared_ptr<core::TensorStorage>& storage
) {

    // Shared tensors may still reference this allocation.
    // Resetting this reference does not invalidate them.
    storage.reset();
}

}
