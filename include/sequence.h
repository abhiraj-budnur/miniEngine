#pragma once

#include "request.h"
#include <vector>
#include <cstdint>

enum class SequenceStatus
{
    WAITING,
    PREFILLING,
    DECODING,
    FINISHED,
    CANCELLED
};

class Sequence
{
public:
    uint64_t id;

    std::vector<int> prompt_tokens;
    std::vector<int> generated_tokens;

    int max_new_tokens;

    int prefill_cursor = 0;
    int current_position = 0;

    int kv_cache_id = -1;

    SequenceStatus status = SequenceStatus::WAITING;

    explicit Sequence(const Request &request);

    bool is_finished() const;

    bool prefill_complete() const;

    int total_tokens() const;

    int remaining_tokens() const;

    void append_token(int token);

    void finish();
};