#pragma once

#include "sequence.h"

#include <memory>
#include <vector>

enum class BatchMode
{
    PREFILL,
    DECODE
};

struct BatchItem
{

    std::shared_ptr<Sequence> sequence;

    int input_token = -1;
    int position = 0;

    int kv_cache_id = -1;
};

struct Batch
{
    BatchMode mode;

    std::vector<BatchItem> items;

    int num_sequences() const
    {
        return static_cast<int>(items.size());
    }

    bool empty() const
    {
        return items.empty();
    }
};