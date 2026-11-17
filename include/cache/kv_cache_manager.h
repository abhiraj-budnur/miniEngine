
#pragma once

#include "cache/kv_cache.h"

#include <memory>
#include <unordered_map>
#include <cstdint>

namespace cache {

class KVCacheManager {

public:

    KVCacheManager(
        int num_layers,
        int num_kv_heads,
        int head_dim,
        int max_sequence_length
    );

    int allocate(uint64_t sequence_id);

    KVCache& get(int cache_id);

    const KVCache& get(int cache_id) const;

    void release(int cache_id);

    bool exists(int cache_id) const;

    size_t total_memory_usage() const;

    size_t active_cache_count() const;

private:

    int num_layers;

    int num_kv_heads;

    int head_dim;

    int max_sequence_length;

    int next_cache_id = 1;

    std::unordered_map<
        int,
        uint64_t
    > cache_to_sequence;

    std::unordered_map<
        int,
        std::unique_ptr<KVCache>
    > caches;

    std::unordered_map<
        uint64_t,
        int
    > sequence_to_cache;
};

}
