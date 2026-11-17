
#include "memory/kv_cache.h"

#include <algorithm>

namespace memory {

KVCache::KVCache(
    size_t num_layers,
    size_t num_kv_heads,
    size_t head_dim,
    size_t max_sequence_length
)
    : num_layers_(num_layers),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim),
      max_sequence_length_(max_sequence_length),
      caches_(num_layers) {

    if (num_layers == 0 ||
        num_kv_heads == 0 ||
        head_dim == 0 ||
        max_sequence_length == 0) {

        throw std::invalid_argument(
            "Invalid KV cache configuration"
        );
    }
}

size_t KVCache::kv_width() const {
    return num_kv_heads_ * head_dim_;
}

void KVCache::append(
    size_t layer,
    uint64_t sequence_id,
    size_t position,
    const float* key,
    const float* value
) {

    if (layer >= num_layers_) {
        throw std::out_of_range("Invalid KV layer");
    }

    if (!key || !value) {
        throw std::invalid_argument("Null KV data");
    }

    if (position >= max_sequence_length_) {
        throw std::out_of_range(
            "KV cache maximum context exceeded"
        );
    }

    auto& sequence = caches_[layer][sequence_id];

    // This reference implementation appends strictly
    // in positional order. It does not support gaps.
    if (position != sequence.length) {
        throw std::runtime_error(
            "KV cache position mismatch"
        );
    }

    const size_t width = kv_width();

    if (sequence.length == sequence.capacity) {

        size_t new_capacity = std::min(
            max_sequence_length_,
            std::max<size_t>(
                16,
                sequence.capacity * 2
            )
        );

        sequence.keys.reserve(
            new_capacity * width
        );

        sequence.values.reserve(
            new_capacity * width
        );

        sequence.capacity = new_capacity;
    }

    sequence.keys.insert(
        sequence.keys.end(),
        key,
        key + width
    );

    sequence.values.insert(
        sequence.values.end(),
        value,
        value + width
    );

    sequence.length++;
}

const SequenceKV& KVCache::get(
    size_t layer,
    uint64_t sequence_id
) const {

    if (layer >= num_layers_) {
        throw std::out_of_range("Invalid KV layer");
    }

    const auto& layer_cache = caches_[layer];

    auto it = layer_cache.find(sequence_id);

    if (it == layer_cache.end()) {
        throw std::runtime_error(
            "Sequence KV cache not found"
        );
    }

    return it->second;
}

void KVCache::clear_sequence(
    uint64_t sequence_id
) {

    for (auto& layer_cache : caches_) {
        layer_cache.erase(sequence_id);
    }
}

void KVCache::clear_layer(size_t layer) {

    if (layer >= num_layers_) {
        throw std::out_of_range("Invalid KV layer");
    }

    caches_[layer].clear();
}

bool KVCache::contains(
    size_t layer,
    uint64_t sequence_id
) const {

    if (layer >= num_layers_) {
        return false;
    }

    return caches_[layer].find(sequence_id) !=
           caches_[layer].end();
}

size_t KVCache::sequence_length(
    size_t layer,
    uint64_t sequence_id
) const {

    if (!contains(layer, sequence_id)) {
        return 0;
    }

    return get(layer, sequence_id).length;
}

size_t KVCache::num_layers() const {
    return num_layers_;
}

size_t KVCache::max_sequence_length() const {
    return max_sequence_length_;
}

}
