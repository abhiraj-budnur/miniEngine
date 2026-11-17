
#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>
#include <stdexcept>

namespace memory {

struct SequenceKV {
    std::vector<float> keys;
    std::vector<float> values;

    size_t length = 0;
    size_t capacity = 0;
};

class KVCache {

public:

    KVCache(
        size_t num_layers,
        size_t num_kv_heads,
        size_t head_dim,
        size_t max_sequence_length
    );

    void append(
        size_t layer,
        uint64_t sequence_id,
        size_t position,
        const float* key,
        const float* value
    );

    const SequenceKV& get(
        size_t layer,
        uint64_t sequence_id
    ) const;

    void clear_sequence(
        uint64_t sequence_id
    );

    void clear_layer(
        size_t layer
    );

    bool contains(
        size_t layer,
        uint64_t sequence_id
    ) const;

    size_t sequence_length(
        size_t layer,
        uint64_t sequence_id
    ) const;

    size_t num_layers() const;
    size_t kv_width() const;
    size_t max_sequence_length() const;

private:

    size_t num_layers_;
    size_t num_kv_heads_;
    size_t head_dim_;
    size_t max_sequence_length_;

    std::vector<
        std::unordered_map<uint64_t, SequenceKV>
    > caches_;

};

}
