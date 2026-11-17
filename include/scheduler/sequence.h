
#pragma once

#include "request.h"

#include <cstdint>
#include <vector>

namespace scheduler {

enum class SequenceStatus {
    WAITING,
    PREFILLING,
    DECODING,
    FINISHED,
    CANCELLED,
    ERROR
};

class Sequence {

public:

    uint64_t id = 0;

    std::vector<int> prompt_tokens;

    std::vector<int> generated_tokens;

    int max_new_tokens = 128;

    int prefill_cursor = 0;

    int kv_cache_id = -1;

    float temperature = 1.0f;

    int top_k = 0;

    float top_p = 1.0f;

    int eos_token_id = -1;

    bool stream = false;

    SequenceStatus status = SequenceStatus::WAITING;

    explicit Sequence(const Request& request);

    bool is_finished() const;

    bool prefill_complete() const;

    int total_tokens() const;

    int remaining_tokens() const;

    int next_decode_position() const;

    void append_generated_token(int token);

    void finish();

    void cancel();

    void fail();
};

}
