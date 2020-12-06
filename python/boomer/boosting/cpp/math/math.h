/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../../common/cpp/data/types.h"


namespace boosting {

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
