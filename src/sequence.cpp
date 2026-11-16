#include "sequence.h"

Sequence::Sequence(const Request &request) : id(request.request_id), prompt_tokens(request.prompt_tokens), max_new_tokens(request.max_new_tokens) {}

bool Sequence::is_finished() const
{
    return status == SequenceStatus::FINISHED ||
           status == SequenceStatus::CANCELLED;
}

bool Sequence::prefill_complete() const
{
    return prefill_cursor >= static_cast<int>(prompt_tokens.size());
}

int Sequence::total_tokens() const
{
    return static_cast<int>(prompt_tokens.size()) + static_cast<int>(generated_tokens.size());
}

int Sequence::remaining_tokens() const
{
    return max_new_tokens - static_cast<int>(generated_tokens.size());
}

void Sequence::append_token(int token)
{
    if (is_finished() || remaining_tokens() <= 0)
        return;

    generated_tokens.push_back(token);

    current_position++;
}

void Sequence::finish()
{
    status = SequenceStatus::FINISHED;
}