
#include "ops/matmul.h"

#include <stdexcept>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

namespace ops {

core::Tensor matmul(
    const core::Tensor& A,
    const core::Tensor& B
) {

    if (A.shape().size() != 2 ||
        B.shape().size() != 2) {

        throw std::invalid_argument(
            "matmul expects two 2D tensors"
        );
    }

    size_t M = A.shape()[0];
    size_t K = A.shape()[1];

    size_t K2 = B.shape()[0];
    size_t N = B.shape()[1];

    if (K != K2) {
        throw std::invalid_argument(
            "matmul dimension mismatch"
        );
    }

    core::Tensor output({M, N});

    if (M == 0 || N == 0 || K == 0) {
        return output;
    }

#ifdef __APPLE__

    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,

        static_cast<int>(M),
        static_cast<int>(N),
        static_cast<int>(K),

        1.0f,

        A.data(),
        static_cast<int>(K),

        B.data(),
        static_cast<int>(N),

        0.0f,

        output.data(),
        static_cast<int>(N)
    );

#else

    // Portable reference implementation.

    for (size_t i = 0; i < M; i++) {

        for (size_t j = 0; j < N; j++) {

            float sum = 0.0f;

            for (size_t k = 0; k < K; k++) {

                sum +=
                    A.data()[i * K + k] *
                    B.data()[k * N + j];
            }

            output.data()[i * N + j] = sum;
        }
    }

#endif

    return output;
}

}
