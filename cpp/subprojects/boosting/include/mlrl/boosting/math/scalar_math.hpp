/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"

namespace math {

    /**
     * Calculates and returns the n-th triangular number, i.e., the number of elements in a n times n triangle.
     *
     * @param n A scalar of type `uint32`, representing the order of the triangular number
     * @return  A scalar of type `uint32`, representing the n-th triangular number
     */
    static inline constexpr uint32 triangularNumber(uint32 n) {
        return (n * (n + 1)) / 2;
    }

    /**
     * Calculates and returns the logistic function `1 / (1 + exp(-x))`, given a specific value `x`.
     *
     * This implementation exploits the identity `1 / (1 + exp(-x)) = exp(x) / (1 + exp(x))` to increase numerical
     * stability (see, e.g., section "Numerically stable sigmoid function" in
     * https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/).
     *
     * @tparam T    The type of the value `x`
     * @param x     The value `x`
     * @return      The value that has been calculated
     */
    template<typename T>
    static inline constexpr T logisticFunction(T x) {
        if (x < 0) {
            T exponential = std::exp(x);  // Evaluates to 0 for large x, resulting in 0 ultimately
            return exponential / (1 + exponential);
        } else {
            T exponential = std::exp(-x);  // Evaluates to 0 for large x, resulting in 1 ultimately
            return 1 / (1 + exponential);
        }
    }

}
