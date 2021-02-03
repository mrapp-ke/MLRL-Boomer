/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../../common/cpp/data/types.h"
#include <cmath>


namespace boosting {

    /**
     * Returns the result of the floating point division `numerator / denominator` or 0, if a division by zero occurs.
     *
     * @tparam T            The type of the operands
     * @param numerator     The numerator
     * @param denominator   The denominator
     * @return              The result of the division or 0, if a division by zero occurred
     */
    template<class T>
    static inline T divideOrZero(T numerator, T denominator) {
        T result = numerator / denominator;
        return std::isfinite(result) ? result : 0;
    }

    /**
     * Allows to compute the mean of several floating point values `x_1, ..., x_n` in an iterative manner, which
     * prevents overflows.
     *
     * This function must be invoked for each value as follows:
     * `mean_1 = iterativeMean(1, x_1, 0); ...; mean_n = iterativeMean(n, x_n, mean_n-1)`
     *
     * @tparam T    The type of the values
     * @param n     The index of the value, starting at 1
     * @param x     The n-th value
     * @return      The mean of all values provided so far
     */
    template<class T>
    static inline T iterativeMean(uint32 n, T x, T mean) {
        return mean + ((x - mean) / (T) n);
    }

    /**
     * Calculates and returns the n-th triangular number, i.e., the number of elements in a n times n triangle.
     *
     * @param n A scalar of type `uint32`, representing the order of the triangular number
     * @return  A scalar of type `uint32`, representing the n-th triangular number
     */
    static inline uint32 triangularNumber(uint32 n) {
        return (n * (n + 1)) / 2;
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
    template<class Iterator>
    static inline float64 l2NormPow(Iterator iterator, uint32 n) {
        float64 result = 0;

        for (uint32 i = 0; i < n; i++) {
            float64 value = iterator[i];
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
    template<class Iterator, class WeightIterator>
    static inline float64 l2NormPow(Iterator iterator, WeightIterator weightIterator, uint32 n) {
        float64 result = 0;

        for (uint32 i = 0; i < n; i++) {
            float64 value = iterator[i];
            float64 weight = (float64) weightIterator[i];
            result += ((value * value) * weight);
        }

        return result;
    }

}
