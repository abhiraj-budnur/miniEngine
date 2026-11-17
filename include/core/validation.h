
#pragma once

#include <cstddef>

namespace validation {

struct ComparisonResult {
    double max_absolute_error = 0.0;
    double mean_squared_error = 0.0;
    double root_mean_squared_error = 0.0;
    double max_relative_error = 0.0;

    std::size_t elements = 0;
    bool passed = false;
};

ComparisonResult compare(
    const float* reference,
    const float* actual,
    std::size_t count,
    double tolerance = 1e-4
);

}
