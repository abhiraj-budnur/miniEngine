
#pragma once

#include "core/tensor.h"

namespace model {

class Linear {

public:

    Linear(
        core::Tensor weight,
        core::Tensor bias = core::Tensor()
    );

    core::Tensor forward(
        const core::Tensor& input
    ) const;

private:

    core::Tensor weight_;
    core::Tensor bias_;

};

}
