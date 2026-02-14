/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"
#include "mlrl/common/util/iterators.hpp"

namespace util {

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

}
