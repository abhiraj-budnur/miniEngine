
#include "scheduler/sequence.h"

#include <algorithm>

namespace scheduler {

Sequence::Sequence(const Request& request)
    : id(request.request_id),
      prompt_tokens(request.prompt_tokens),
      max_new_tokens(request.max_new_tokens),
      temperature(request.temperature),
      top_k(request.top_k),
      top_p(request.top_p),
      eos_token_id(request.eos_token_id),
      stream(request.stream) {
}

bool Sequence::is_finished() const {

    return status == SequenceStatus::FINISHED ||
           status == SequenceStatus::CANCELLED ||
           status == SequenceStatus::ERROR;
}

bool Sequence::prefill_complete() const {

    return prefill_cursor >=
           static_cast<int>(prompt_tokens.size());
}

int Sequence::total_tokens() const {

    return static_cast<int>(prompt_tokens.size()) +
           static_cast<int>(generated_tokens.size());
}

int Sequence::remaining_tokens() const {

    return std::max(
        0,
        max_new_tokens -
        static_cast<int>(generated_tokens.size())
    );
}

int Sequence::next_decode_position() const {

    // The last generated token has not yet been
    // consumed by the model at this point.
    return static_cast<int>(prompt_tokens.size()) +
           static_cast<int>(generated_tokens.size()) - 1;
}

void Sequence::append_generated_token(int token) {

    if (is_finished() || status != SequenceStatus::DECODING) {
        return;
    }

    if (remaining_tokens() <= 0) {
        finish();
        return;
    }

    generated_tokens.push_back(token);

    if (token == eos_token_id ||
        remaining_tokens() <= 0) {
        finish();
    }
}

void Sequence::finish() {
    status = SequenceStatus::FINISHED;
}

void Sequence::cancel() {
    status = SequenceStatus::CANCELLED;
}

void Sequence::fail() {
    status = SequenceStatus::ERROR;
}

}
