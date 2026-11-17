
#include "cache/kv_cache.h"

#include <stdexcept>
#include <limits>

namespace cache {

KVCache::KVCache(
    int num_layers,
    int num_kv_heads,
    int head_dim,
    int max_sequence_length
)
    : layers(num_layers),
      kv_heads(num_kv_heads),
      dimension(head_dim),
      max_length(max_sequence_length) {

    if (layers <= 0 ||
        kv_heads <= 0 ||
        dimension <= 0 ||
        max_length <= 0) {

        throw std::invalid_argument(
            "Invalid KV cache dimensions"
        );
    }

    size_t elements = static_cast<size_t>(layers);

    const size_t dimensions[] = {
        static_cast<size_t>(max_length),
        static_cast<size_t>(kv_heads),
        static_cast<size_t>(dimension)
    };

    for (size_t d : dimensions) {
        if (elements >
            std::numeric_limits<size_t>::max() / d) {

            throw std::overflow_error(
                "KV cache size overflow"
            );
        }

        elements *= d;
    }

    key_storage.resize(elements, 0.0f);

    value_storage.resize(elements, 0.0f);

    lengths.resize(layers, 0);
}

size_t KVCache::offset(
    int layer,
    int position,
    int head,
    int dim
) const {

    return (
        (
            (
                static_cast<size_t>(layer) * max_length +
                position
            ) * kv_heads +
            head
        ) * dimension
    ) + dim;
}

void KVCache::append(
    int layer,
    const float* keys,
    const float* values,
    int position
) {

    if (layer < 0 || layer >= layers) {
        throw std::out_of_range("Invalid KV layer");
    }

    if (!keys || !values) {
        throw std::invalid_argument(
            "Null KV input"
        );
    }

    if (position < 0 || position >= max_length) {
        throw std::out_of_range(
            "KV cache capacity exceeded"
        );
    }

    if (position != lengths[layer]) {
        throw std::runtime_error(
            "KV cache position is not sequential"
        );
    }

    for (int h = 0; h < kv_heads; h++) {

        for (int d = 0; d < dimension; d++) {

            size_t destination =
                offset(layer, position, h, d);

            size_t source =
                static_cast<size_t>(h) * dimension + d;

            key_storage[destination] = keys[source];

            value_storage[destination] = values[source];
        }
    }

    lengths[layer]++;
}

const float* KVCache::get_keys(int layer) const {

    if (layer < 0 || layer >= layers) {
        throw std::out_of_range("Invalid KV layer");
    }

    return key_storage.data() +
        static_cast<size_t>(layer) *
        max_length * kv_heads * dimension;
}

const float* KVCache::get_values(int layer) const {

    if (layer < 0 || layer >= layers) {
        throw std::out_of_range("Invalid KV layer");
    }

    return value_storage.data() +
        static_cast<size_t>(layer) *
        max_length * kv_heads * dimension;
}

int KVCache::current_length() const {

    if (lengths.empty()) {
        return 0;
    }

    return lengths.front();
}

int KVCache::capacity() const {
    return max_length;
}

int KVCache::num_kv_heads() const {
    return kv_heads;
}

int KVCache::head_dim() const {
    return dimension;
}

size_t KVCache::memory_usage() const {

    return (
        key_storage.size() +
        value_storage.size()
    ) * sizeof(float);
}

}
