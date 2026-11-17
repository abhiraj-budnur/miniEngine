
#pragma once

#include "batch.h"
#include "request.h"
#include "sequence.h"

#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>
#include <cstdint>

namespace scheduler {

struct SchedulerConfig {

    int max_num_sequences = 8;

    int max_num_batched_tokens = 1024;

    int max_prefill_chunk_size = 256;
};

class Scheduler {

private:

    SchedulerConfig config;

    std::deque<std::shared_ptr<Sequence>> waiting_queue;

    std::vector<std::shared_ptr<Sequence>> active_sequences;

    std::unordered_map<
        uint64_t,
        std::shared_ptr<Sequence>
    > all_sequences;

    bool prefer_prefill = true;

public:

    explicit Scheduler(const SchedulerConfig& config);

    void add_request(const Request& request);

    Batch schedule();

    // Called only after successful model execution.
    void advance_prefill(
        uint64_t sequence_id,
        int tokens_processed
    );

    void append_generated_token(
        uint64_t sequence_id,
        int token
    );

    void complete_sequence(uint64_t sequence_id);

    void cancel_sequence(uint64_t sequence_id);

    void fail_sequence(uint64_t sequence_id);

    std::shared_ptr<Sequence> get_sequence(
        uint64_t sequence_id
    ) const;

    bool has_pending_requests() const;

    int active_count() const;

    int waiting_count() const;

private:

    void remove_finished_sequences();

    void admit_waiting_sequences();

    Batch build_prefill_batch();

    Batch build_decode_batch();

    bool has_prefill_work() const;

    bool has_decode_work() const;
};

}
