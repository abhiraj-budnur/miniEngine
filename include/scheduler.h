#pragma once

#include "batch.h"
#include "sequence.h"

#include <deque>
#include <memory>
#include <vector>
#include <unordered_map>

struct SchedulerConfig
{
    int max_num_sequences = 8;
    int max_batched_tokens = 1024;
    int max_prefill_chunk_size = 256;
};

class Scheduler
{
private:
    SchedulerConfig config;

    std::deque<std::shared_ptr<Sequence>> waiting_queue;

    std::deque<std::shared_ptr<Sequence>> active_sequences;

    std::unordered_map<uint64_t, std::shared_ptr<Sequence>> all_sequences;

public:
    explicit Scheduler(const SchedulerConfig &config);

    void add_request(std::shared_ptr<Sequence> sequence);

    Batch schedule();

    void complete_sequence(uint64_t sequence_id);

    void cancel_sequence(uint64_t sequence_id);

    bool has_pending_requests() const;

    int active_count() const;

    int waiting_count() const;

private:
    void remove_finished_sequences();

    void admit_waiting_sequences();

    Batch build_prefill_batch();

    Batch build_decode_batch();
};