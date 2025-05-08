/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"
#include "mlrl/common/util/iterators.hpp"

namespace util {

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
     * Computes and returns the L1 norm of a specific vector, i.e., the sum of the absolute values of its elements.
     *
     * @tparam Iterator The type of the iterator that provides access to the elements in the vector
     * @param iterator  An iterator of template type `Iterator` that provides random access to the elements in the
     *                  vector
     * @param n         The number of elements in the vector
     * @return          The L1 norm
     */
    template<typename Iterator>
    static inline constexpr typename util::iterator_value<Iterator> l1Norm(Iterator iterator, uint32 n) {
        typename util::iterator_value<Iterator> result = 0;

        for (uint32 i = 0; i < n; i++) {
            result += std::abs(iterator[i]);
        }

        return result;
    }

    /**
     * Computes and returns the L1 norm of a specific vector, i.e., the sum of the absolute values of its elements,
     * where each element has a specific weight.
     *
     * @tparam Iterator         The type of the iterator that provides access to the elements in the vector
     * @tparam WeightIterator   The type of the iterator that provides access to the weights of the elements
     * @param iterator          An iterator of template type `Iterator` that provides random access to the elements in
     *                          the vector
     * @param weightIterator    An iterator of template type `WeightIterator` that provides random access to the weights
     *                          of the elements
     * @param n                 The number of elements in the vector
     * @return                  The L1 norm
     */
    template<typename Iterator, typename WeightIterator>
    static inline constexpr typename util::iterator_value<Iterator> l1Norm(Iterator iterator,
                                                                           WeightIterator weightIterator, uint32 n) {
        typename util::iterator_value<Iterator> result = 0;

        for (uint32 i = 0; i < n; i++) {
            result += (std::abs(iterator[i]) * weightIterator[i]);
        }

        return result;
    }

    /**
     * Computes and returns the square of the L2 norm of a specific vector, i.e. the sum of the squares of its elements.
     * To obtain the actual L2 norm, the square-root of the result provided by this function must be computed.
     *
     * @tparam Iterator The type of the iterator that provides access to the elements in the vector
     * @param iterator  An iterator of template type `Iterator` that provides random access to the elements in the
     *                  vector
     * @param n         The number of elements in the vector
     * @return          The square of the L2 norm
     */
    template<typename Iterator>
    static inline constexpr typename util::iterator_value<Iterator> l2NormPow(Iterator iterator, uint32 n) {
        typename util::iterator_value<Iterator> result = 0;

        for (uint32 i = 0; i < n; i++) {
            typename util::iterator_value<Iterator> value = iterator[i];
            result += (value * value);
        }

        return result;
    }

    /**
     * Computes and returns the square of the L2 norm of a specific vector, i.e. the sum of the squares of its elements,
     * where each elements has a specific weight. To obtain the actual L2 norm, the square-root of the result provided
     * by this function must be computed.
     *
     * @tparam Iterator         The type of the iterator that provides access to the elements in the vector
     * @tparam WeightIterator   The type of the iterator that provides access to the weights of the elements
     * @param iterator          An iterator of template type `Iterator` that provides random access to the elements in
     *                          the vector
     * @param weightIterator    An iterator of template type `WeightIterator` that provides random access to the weights
     *                          of the elements
     * @param n                 The number of elements in the vector
     * @return                  The square of the L2 norm
     */
    template<typename Iterator, typename WeightIterator>
    static inline constexpr typename util::iterator_value<Iterator> l2NormPow(Iterator iterator,
                                                                              WeightIterator weightIterator, uint32 n) {
        typename util::iterator_value<Iterator> result = 0;

        for (uint32 i = 0; i < n; i++) {
            typename util::iterator_value<Iterator> value = iterator[i];
            result += ((value * value) * weightIterator[i]);
        }

        return result;
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
