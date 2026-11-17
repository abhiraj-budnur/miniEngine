
#pragma once

#include <cstddef>
#include <vector>

namespace cache {

class KVCache {

public:

    KVCache(
        int num_layers,
        int num_kv_heads,
        int head_dim,
        int max_sequence_length
    );

    void append(
        int layer,
        const float* keys,
        const float* values,
        int position
    );

    const float* get_keys(int layer) const;

    const float* get_values(int layer) const;

    int current_length() const;

    int capacity() const;

    int num_kv_heads() const;

    int head_dim() const;

    size_t memory_usage() const;

private:

    int layers;

    int kv_heads;

    int dimension;

    int max_length;

    std::vector<float> key_storage;

    std::vector<float> value_storage;

    // One logical sequence length per layer.
    std::vector<int> lengths;

    size_t offset(
        int layer,
        int position,
        int head,
        int dimension
    ) const;
};

}
