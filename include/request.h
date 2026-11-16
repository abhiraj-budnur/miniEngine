#pragma once

#include <cstdint>
#include <vector>

struct Request{
    uint64_t request_id;

    std::vector<int> prompt_tokens;

    int max_new_tokens = 128;

    float temperature = 1.0;

    bool stream = false;
};