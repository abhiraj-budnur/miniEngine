#include "scheduler.h"

#include <algorithm>
#include <stdexcept>

Scheduler::Scheduler(const SchedulerConfig &cfg) : config(cfg)
{
}

void Scheduler::add_request(
    std::shared_ptr<Sequence> sequence)
{
    if (!sequence)
    {
        throw std::invalid_argument("Null sequence");
    }
    if (all_sequences.count(sequence->id))
    {
        throw std::invalid_argument("Duplicate Request ID");
    }

    sequence->status = SequenceStatus::WAITING;

    waiting_queue.push_back(sequence);

    all_sequences[sequence->id] = sequence;
}

void Scheduler::remove_finished_sequences()
{
    active_sequences.erase(
        std::remove_if(
            active_sequences.begin(),
            active_sequences.end(),
            [](const auto &seq)
            {
                return seq->is_finished();
            }),
        active_sequences.end());

    for (auto it = all_sequences.begin(); it != all_sequences.end();)
    {
        if (it->second->is_finished())
        {
            it = all_sequences.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

void Scheduler::admit_waiting_sequences()
{
    while (!waiting_queue.empty() && static_cast<int>(active_sequences.size()) < config.max_num_sequences)
    {
        auto sequence = waiting_queue.front();

        waiting_queue.pop_front();

        sequence->status = SequenceStatus::PREFILLING;

        active_sequences.push_back(sequence);
    }
}

Batch Scheduler::build_prefill_batch()
{

    Batch batch;

    batch.mode = BatchMode::PREFILL;

    int remaining_budget = config.max_batched_tokens;

    for (auto &sequence : active_sequences)
    {
        if (sequence->status != SequenceStatus::PREFILLING)
        {
            continue;
        }

        if (remaining_budget <= 0)
            break;

        int remaining_prompt = static_cast<int>(sequence->prompt_tokens.size()) - sequence->prefill_cursor;

        int chunk_size = std::min({remaining_prompt,
                                   config.max_prefill_chunk_size,
                                   remaining_budget});

        if (chunk_size <= 0)
            continue;

        BatchItem item;

        item.sequence = sequence;
        item.position = sequence->prefill_cursor;
        item.kv_cache_id = sequence->kv_cache_id;

        batch.items.push_back(item);

        remaining_budget -= chunk_size;
    }
    return batch;
}

Batch Scheduler::build_decode_batch(){
    Batch batch;

    batch.mode = BatchMode::DECODE;

    int budget = config.max_batched_tokens;

    for(auto& sequence:active_sequences){
        
        if(sequence->status!=SequenceStatus::DECODING)continue;
        
        if(budget<=0)break;
        
        BatchItem item;

        item.sequence = sequence;

        item.input_token = sequence->generated_tokens.back();

        item.position = sequence ->current_position;

        item.kv_cache_id = sequence->kv_cache_id;

        batch.items.push_back(item);

        budget--;
    }
    return batch;
}

Batch Scheduler::schedule(){
    remove_finished_sequences();
    admit_waiting_sequences();

    Batch decode_batch = build_decode_batch();

    if(!decode_batch.empty())return decode_batch;

    return build_prefill_batch();
}

void Scheduler::complete_sequence(uint64_t)