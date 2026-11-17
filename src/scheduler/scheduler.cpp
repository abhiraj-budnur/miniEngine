
#include "scheduler/scheduler.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace scheduler {

Scheduler::Scheduler(const SchedulerConfig& cfg)
    : config(cfg) {

    if (config.max_num_sequences <= 0 ||
        config.max_num_batched_tokens <= 0 ||
        config.max_prefill_chunk_size <= 0) {

        throw std::invalid_argument(
            "Invalid scheduler configuration"
        );
    }
}

void Scheduler::add_request(const Request& request) {

    if (request.request_id == 0) {
        throw std::invalid_argument(
            "Request ID cannot be zero"
        );
    }

    if (request.prompt_tokens.empty()) {
        throw std::invalid_argument(
            "Prompt cannot be empty"
        );
    }

    if (request.max_new_tokens <= 0) {
        throw std::invalid_argument(
            "max_new_tokens must be positive"
        );
    }

    if (request.temperature < 0.0f ||
        request.top_p <= 0.0f ||
        request.top_p > 1.0f ||
        request.top_k < 0) {

        throw std::invalid_argument(
            "Invalid sampling configuration"
        );
    }

    if (all_sequences.count(request.request_id)) {
        throw std::invalid_argument(
            "Duplicate request ID"
        );
    }

    auto sequence = std::make_shared<Sequence>(request);

    sequence->status = SequenceStatus::WAITING;

    waiting_queue.push_back(sequence);

    all_sequences[sequence->id] = sequence;
}

void Scheduler::remove_finished_sequences() {

    active_sequences.erase(
        std::remove_if(
            active_sequences.begin(),
            active_sequences.end(),
            [](const auto& seq) {
                return seq->is_finished();
            }
        ),
        active_sequences.end()
    );

    for (auto it = all_sequences.begin();
         it != all_sequences.end();) {

        if (it->second->is_finished()) {
            it = all_sequences.erase(it);
        } else {
            ++it;
        }
    }

    // Remove cancelled or failed requests from waiting queue.
    waiting_queue.erase(
        std::remove_if(
            waiting_queue.begin(),
            waiting_queue.end(),
            [](const auto& seq) {
                return seq->is_finished();
            }
        ),
        waiting_queue.end()
    );
}

void Scheduler::admit_waiting_sequences() {

    while (
        !waiting_queue.empty() &&
        static_cast<int>(active_sequences.size()) <
            config.max_num_sequences
    ) {

        auto sequence = waiting_queue.front();

        waiting_queue.pop_front();

        if (sequence->is_finished()) {
            continue;
        }

        sequence->status = SequenceStatus::PREFILLING;

        active_sequences.push_back(sequence);
    }
}

bool Scheduler::has_prefill_work() const {

    for (const auto& seq : active_sequences) {

        if (seq->status == SequenceStatus::PREFILLING) {
            return true;
        }
    }

    return false;
}

bool Scheduler::has_decode_work() const {

    for (const auto& seq : active_sequences) {

        if (seq->status == SequenceStatus::DECODING &&
            !seq->generated_tokens.empty()) {

            return true;
        }
    }

    return false;
}

Batch Scheduler::build_prefill_batch() {

    Batch batch;

    batch.mode = BatchMode::PREFILL;

    int remaining_budget =
        config.max_num_batched_tokens;

    for (auto& sequence : active_sequences) {

        if (sequence->status != SequenceStatus::PREFILLING) {
            continue;
        }

        if (remaining_budget <= 0) {
            break;
        }

        int remaining_prompt =
            static_cast<int>(sequence->prompt_tokens.size()) -
            sequence->prefill_cursor;

        int chunk_size = std::min({
            remaining_prompt,
            config.max_prefill_chunk_size,
            remaining_budget
        });

        if (chunk_size <= 0) {
            continue;
        }

        BatchItem item;

        item.sequence = sequence;

        item.position = sequence->prefill_cursor;

        item.kv_cache_id = sequence->kv_cache_id;

        auto begin = sequence->prompt_tokens.begin() +
                     sequence->prefill_cursor;

        auto end = begin + chunk_size;

        item.input_tokens.assign(begin, end);

        batch.items.push_back(std::move(item));

        remaining_budget -= chunk_size;
    }

    return batch;
}

Batch Scheduler::build_decode_batch() {

    Batch batch;

    batch.mode = BatchMode::DECODE;

    int remaining_budget =
        config.max_num_batched_tokens;

    for (auto& sequence : active_sequences) {

        if (sequence->status != SequenceStatus::DECODING) {
            continue;
        }

        if (remaining_budget <= 0) {
            break;
        }

        if (sequence->generated_tokens.empty()) {
            continue;
        }

        BatchItem item;

        item.sequence = sequence;

        item.input_tokens.push_back(
            sequence->generated_tokens.back()
        );

        item.position = sequence->next_decode_position();

        item.kv_cache_id = sequence->kv_cache_id;

        batch.items.push_back(std::move(item));

        remaining_budget--;
    }

    return batch;
}

Batch Scheduler::schedule() {

    // 1. Remove completed requests.
    remove_finished_sequences();

    // 2. Admit new requests.
    admit_waiting_sequences();

    bool prefill_available = has_prefill_work();

    bool decode_available = has_decode_work();

    // 3. Select work without permanently starving prefill.
    if (prefill_available && decode_available) {

        if (prefer_prefill) {

            prefer_prefill = false;

            return build_prefill_batch();
        }

        prefer_prefill = true;

        return build_decode_batch();
    }

    if (decode_available) {
        return build_decode_batch();
    }

    if (prefill_available) {
        return build_prefill_batch();
    }

    return Batch{};
}

void Scheduler::advance_prefill(
    uint64_t sequence_id,
    int tokens_processed
) {

    auto sequence = get_sequence(sequence_id);

    if (!sequence) {
        throw std::runtime_error(
            "Sequence not found"
        );
    }

    if (sequence->status != SequenceStatus::PREFILLING) {
        throw std::runtime_error(
            "Sequence is not prefilling"
        );
    }

    if (tokens_processed <= 0) {
        throw std::invalid_argument(
            "Invalid prefill progress"
        );
    }

    int remaining =
        static_cast<int>(sequence->prompt_tokens.size()) -
        sequence->prefill_cursor;

    if (tokens_processed > remaining) {
        throw std::invalid_argument(
            "Prefill progress exceeds remaining prompt"
        );
    }

    sequence->prefill_cursor += tokens_processed;

    // Once prefill is complete, the inference engine
    // must sample the first token and append it.
    //
    // append_generated_token() changes status to DECODING.
}

void Scheduler::append_generated_token(
    uint64_t sequence_id,
    int token
) {

    auto sequence = get_sequence(sequence_id);

    if (!sequence) {
        throw std::runtime_error(
            "Sequence not found"
        );
    }

    if (!sequence->prefill_complete()) {
        throw std::runtime_error(
            "Cannot generate before prefill completes"
        );
    }

    if (sequence->status == SequenceStatus::PREFILLING) {
        sequence->status = SequenceStatus::DECODING;
    }

    if (sequence->status != SequenceStatus::DECODING) {
        throw std::runtime_error(
            "Sequence is not ready for decoding"
        );
    }

    sequence->append_generated_token(token);
}

void Scheduler::complete_sequence(
    uint64_t sequence_id
) {

    auto sequence = get_sequence(sequence_id);

    if (sequence) {
        sequence->finish();
    }
}

void Scheduler::cancel_sequence(
    uint64_t sequence_id
) {

    auto sequence = get_sequence(sequence_id);

    if (sequence) {
        sequence->cancel();
    }
}

void Scheduler::fail_sequence(
    uint64_t sequence_id
) {

    auto sequence = get_sequence(sequence_id);

    if (sequence) {
        sequence->fail();
    }
}

std::shared_ptr<Sequence> Scheduler::get_sequence(
    uint64_t sequence_id
) const {

    auto it = all_sequences.find(sequence_id);

    if (it == all_sequences.end()) {
        return nullptr;
    }

    return it->second;
}

bool Scheduler::has_pending_requests() const {

    return !waiting_queue.empty() ||
           !active_sequences.empty();
}

int Scheduler::active_count() const {

    return static_cast<int>(
        active_sequences.size()
    );
}

int Scheduler::waiting_count() const {

    return static_cast<int>(
        waiting_queue.size()
    );
}

}
