
#pragma once

#include "sequence.h"

#include <memory>
#include <vector>

namespace scheduler {

enum class BatchMode {
    PREFILL,
    DECODE
};

struct BatchItem {

    std::shared_ptr<Sequence> sequence;

    std::vector<int> input_tokens;

    int position = 0;

    int kv_cache_id = -1;

    int num_tokens() const {
        return static_cast<int>(input_tokens.size());
    }
};

struct Batch {

    BatchMode mode = BatchMode::PREFILL;

    std::vector<BatchItem> items;

    int num_sequences() const {
        return static_cast<int>(items.size());
    }

    int num_tokens() const {

        int count = 0;

        for (const auto& item : items) {
            count += item.num_tokens();
        }

        return count;
    }

    bool empty() const {
        return items.empty();
    }
};

}
