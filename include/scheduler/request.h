
#pragma once

#include <cstdint>
#include <vector>

namespace scheduler {

struct Request {

    uint64_t request_id = 0;

    std::vector<int> prompt_tokens;

    int max_new_tokens = 128;

    float temperature = 1.0f;

    int top_k = 0;

    float top_p = 1.0f;

    int eos_token_id = -1;

    bool stream = false;
};

}
