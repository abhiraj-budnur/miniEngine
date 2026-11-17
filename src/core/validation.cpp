
#include "core/validation.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace validation {

ComparisonResult compare(
    const float* reference,
    const float* actual,
    std::size_t count,
    double tolerance
) {
    if (!reference || !actual || count == 0) {
        throw std::invalid_argument(
            "Invalid comparison inputs"
        );
    }

    ComparisonResult result;
    result.elements = count;

    double squared_error = 0.0;
    double max_relative = 0.0;

    for (std::size_t i = 0; i < count; ++i) {

        if (!std::isfinite(reference[i]) ||
            !std::isfinite(actual[i])) {
            throw std::runtime_error(
                "Non-finite value encountered"
            );
        }

        double diff = std::abs(
            static_cast<double>(reference[i]) -
            static_cast<double>(actual[i])
        );

        result.max_absolute_error =
            std::max(result.max_absolute_error, diff);

        squared_error += diff * diff;

        double denominator = std::max(
            std::abs(static_cast<double>(reference[i])),
            1e-8
        );

        max_relative = std::max(
            max_relative,
            diff / denominator
        );
    }

    result.mean_squared_error =
        squared_error / count;

    result.root_mean_squared_error =
        std::sqrt(result.mean_squared_error);

    result.max_relative_error = max_relative;

    result.passed =
        result.max_absolute_error <= tolerance;

    return result;
}

}
