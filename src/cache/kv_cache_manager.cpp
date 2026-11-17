
#include "cache/kv_cache_manager.h"

#include <stdexcept>
#include <limits>

namespace cache {

KVCacheManager::KVCacheManager(
    int layers,
    int kv_heads,
    int head_dim,
    int max_sequence_length
)
    : num_layers(layers),
      num_kv_heads(kv_heads),
      head_dim(head_dim),
      max_sequence_length(max_sequence_length) {
}

int KVCacheManager::allocate(uint64_t sequence_id) {

    if (sequence_id == 0) {
        throw std::invalid_argument(
            "Invalid sequence ID"
        );
    }

    if (sequence_to_cache.count(sequence_id)) {
        throw std::runtime_error(
            "Sequence already owns a KV cache"
        );
    }

    if (next_cache_id == std::numeric_limits<int>::max()) {
        throw std::overflow_error(
            "KV cache ID space exhausted"
        );
    }

    int cache_id = next_cache_id++;

    auto cache = std::make_unique<KVCache>(
        num_layers,
        num_kv_heads,
        head_dim,
        max_sequence_length
    );

    caches.emplace(cache_id, std::move(cache));

    cache_to_sequence[cache_id] = sequence_id;

    sequence_to_cache[sequence_id] = cache_id;

    return cache_id;
}

KVCache& KVCacheManager::get(int cache_id) {

    auto it = caches.find(cache_id);

    if (it == caches.end()) {
        throw std::runtime_error(
            "KV cache not found"
        );
    }

    return *it->second;
}

const KVCache& KVCacheManager::get(int cache_id) const {

    auto it = caches.find(cache_id);

    if (it == caches.end()) {
        throw std::runtime_error(
            "KV cache not found"
        );
    }

    return *it->second;
}

void KVCacheManager::release(int cache_id) {

    auto it = caches.find(cache_id);

    if (it == caches.end()) {
        return;
    }

    auto sequence_it = cache_to_sequence.find(cache_id);

    if (sequence_it != cache_to_sequence.end()) {

        sequence_to_cache.erase(sequence_it->second);

        cache_to_sequence.erase(sequence_it);
    }

    caches.erase(it);
}

bool KVCacheManager::exists(int cache_id) const {
    return caches.count(cache_id) != 0;
}

size_t KVCacheManager::total_memory_usage() const {

    size_t total = 0;

    for (const auto& entry : caches) {
        total += entry.second->memory_usage();
    }

    return total;
}

size_t KVCacheManager::active_cache_count() const {
    return caches.size();
}

}
