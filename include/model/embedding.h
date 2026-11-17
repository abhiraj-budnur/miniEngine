
#pragma once

#include "core/tensor.h"

#include <cstdint>
#include <vector>

namespace model {

class Embedding {

public:

    explicit Embedding(
        core::Tensor weight
    );

    core::Tensor forward(
        const std::vector<int32_t>& token_ids
    ) const;

private:

    core::Tensor weight_;

};

}
