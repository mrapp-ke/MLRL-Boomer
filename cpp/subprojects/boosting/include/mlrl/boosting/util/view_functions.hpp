/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"

#include <algorithm>

namespace boosting {

    /**
     * Adds the elements in a view `b` to the elements in another view `a`, such that `a = a + b`.
     *
     * @tparam IteratorA    The type of the iterator that provides access to the values in the view `a`
     * @tparam IteratorB    The type of the iterator that provides access to the values in the view `b`
     * @param a             An iterator to the beginning of the view `a`
     * @param b             An iterator to the beginning of the view `b`
     * @param numElements   The number of elements in the views `a` and `b`
     *
     */
    template<typename IteratorA, typename IteratorB>
    static inline void addToView(IteratorA a, IteratorB b, uint32 numElements) {
        for (uint32 i = 0; i < numElements; i++) {
            a[i] += b[i];
        }
    }

    /**
     * Adds the elements in a view `b` to the elements in another view `a`. The elements in the view `b` are multiplied
     * by a given weight, such that `a = a + (b * weight)`.
     *
     * @tparam IteratorA    The type of the iterator that provides access to the values in the view `a`
     * @tparam IteratorB    The type of the iterator that provides access to the values in the view `b`
     * @tparam Weight       The type of the weight
     * @param a             An iterator to the beginning of the view `a`
     * @param b             An iterator to the beginning of the view `b`
     * @param numElements   The number of elements in the views `a` and `b`
     * @param weight        The weight, the elements in the view `b` should be multiplied by
     *
     */
    template<typename IteratorA, typename IteratorB, typename Weight>
    static inline void addToView(IteratorA a, IteratorB b, uint32 numElements, Weight weight) {
        for (uint32 i = 0; i < numElements; i++) {
            a[i] += (b[i] * weight);
        }
    }

    /**
     * Adds the elements in a view `b` to the elements in another view `a`, such that `a = a + b`. The indices of the
     * elements in the view `b` that correspond to the elements in the view `a` are given as an additional view.
     *
     * @tparam IteratorA        The type of the iterator that provides access to the values in the view `a`
     * @tparam IteratorB        The type of the iterator that provides access to the values in the view `b`
     * @tparam IndexIterator    The type of the iterator that provides access to the indices
     * @param a                 An iterator to the beginning of the view `a`
     * @param b                 An iterator to the beginning of the view `b`
     * @param numElements       The number of elements in the views `a` and `b`
     * @param indices           An iterator to the beginning of a view that stores the indices of the elements in the
     *                          view `b` that correspond to the elements in the view `a`
     *
     */
    template<typename IteratorA, typename IteratorB, typename IndexIterator>
    static inline void addToView(IteratorA a, IteratorB b, IndexIterator indices, uint32 numElements) {
        for (uint32 i = 0; i < numElements; i++) {
            uint32 index = indices[i];
            a[i] += b[index];
        }
    }

    /**
     * Adds the elements in a view `b` to the elements in another view `a`. The elements in the view `b` are multiplied
     * by a given weight, such that `a = a + (b * weight)`. The indices of the elements in the view `b` that correspond
     * to the elements in the view `a` are given as an additional view.
     *
     * @tparam IteratorA        The type of the iterator that provides access to the values in the view `a`
     * @tparam IteratorB        The type of the iterator that provides access to the values in the view `b`
     * @tparam IndexIterator    The type of the iterator that provides access to the indices
     * @tparam Weight           The type of the weight
     * @param a                 An iterator to the beginning of the view `a`
     * @param b                 An iterator to the beginning of the view `b`
     * @param numElements       The number of elements in the views `a` and `b`
     * @param weight            The weight, the elements in the view `b` should be multiplied by
     * @param indices           An iterator to the beginning of a view that stores the indices of the elements in the
     *                          view `b` that correspond to the elements in the view `a`
     *
     */
    template<typename IteratorA, typename IteratorB, typename IndexIterator, typename Weight>
    static inline void addToView(IteratorA a, IteratorB b, IndexIterator indices, uint32 numElements, Weight weight) {
        for (uint32 i = 0; i < numElements; i++) {
            uint32 index = indices[i];
            a[i] += (b[index] * weight);
        }
    }

    /**
     * Removes the elements in a view `b` from the elements in another view `a`, such that `a = a - b`.
     *
     * @tparam IteratorA    The type of the iterator that provides access to the values in the view `a`
     * @tparam IteratorB    The type of the iterator that provides access to the values in the view `b`
     * @param a             An iterator to the beginning of the view `a`
     * @param b             An iterator to the beginning of the view `b`
     * @param numElements   The number of elements in the views `a` and `b`
     */
    template<typename IteratorA, typename IteratorB>
    static inline void removeFromView(IteratorA a, IteratorB b, uint32 numElements) {
        for (uint32 i = 0; i < numElements; i++) {
            a[i] -= b[i];
        }
    }

    /**
     * Removes the elements in a view `b` from the elements in another view `a`. The elements in the view `b` are
     * multiplied by a given weight, such that `a = a - (b * weight)`.
     *
     * @tparam IteratorA    The type of the iterator that provides access to the values in the view `a`
     * @tparam IteratorB    The type of the iterator that provides access to the values in the view `b`
     * @tparam Weight       The type of the weight
     * @param a             An iterator to the beginning of the view `a`
     * @param b             An iterator to the beginning of the view `b`
     * @param numElements   The number of elements in the views `a` and `b`
     * @param weight        The weight, the elements in the view `b` should be multiplied by
     *
     */
    template<typename IteratorA, typename IteratorB, typename Weight>
    static inline void removeFromView(IteratorA a, IteratorB b, uint32 numElements, Weight weight) {
        for (uint32 i = 0; i < numElements; i++) {
            a[i] -= (b[i] * weight);
        }
    }

}
