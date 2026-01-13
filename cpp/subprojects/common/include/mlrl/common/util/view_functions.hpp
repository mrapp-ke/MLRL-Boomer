/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"

namespace util {

    /**
     * Copy all elements from one view to another.
     *
     * @tparam FromIterator The type of the iterator that provides access to the values in the view to copy from
     * @tparam ToIterator   The type of the iterator that provides access to the values in the view to copy to
     * @param from          An iterator to the beginning of the view to copy from
     * @param to            An iterator to the beginning of the view to copy to
     * @param numElements   The number of elements to be copied
     */
    template<typename FromIterator, typename ToIterator>
    static inline void copy(FromIterator from, ToIterator to, uint32 numElements) {
        for (uint32 i = 0; i < numElements; i++) {
            to[i] = from[i];
        }
    }

    /**
     * Adds the elements in a view `b` to the elements in another view `a`, such that `a = a + b`.
     *
     * @tparam IteratorA    The type of the iterator that provides access to the values in the view `a`
     * @tparam IteratorB    The type of the iterator that provides access to the values in the view `b`
     * @param a             An iterator to the beginning of the view `a`
     * @param b             An iterator to the beginning of the view `b`
     * @param numElements   The number of elements in the views `a` and `b`
     */
    template<typename IteratorA, typename IteratorB>
    static inline void add(IteratorA a, IteratorB b, uint32 numElements) {
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
     */
    template<typename IteratorA, typename IteratorB, typename Weight>
    static inline void addWeighted(IteratorA a, IteratorB b, uint32 numElements, Weight weight) {
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
     */
    template<typename IteratorA, typename IteratorB, typename IndexIterator>
    static inline void add(IteratorA a, IteratorB b, IndexIterator indices, uint32 numElements) {
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
     */
    template<typename IteratorA, typename IteratorB, typename IndexIterator, typename Weight>
    static inline void addWeighted(IteratorA a, IteratorB b, IndexIterator indices, uint32 numElements, Weight weight) {
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
    static inline void subtract(IteratorA a, IteratorB b, uint32 numElements) {
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
     */
    template<typename IteratorA, typename IteratorB, typename Weight>
    static inline void subtractWeighted(IteratorA a, IteratorB b, uint32 numElements, Weight weight) {
        for (uint32 i = 0; i < numElements; i++) {
            a[i] -= (b[i] * weight);
        }
    }

    /**
     * Sets all elements in a view `a` to the difference between the elements in two other views `b` and `c`, such that
     * `a = b - c`.
     *
     * @tparam IteratorA    The type of the iterator that provides access to the value in the view `a`
     * @tparam IteratorB    The type of the iterator that provides access to the value in the view `b`
     * @tparam IteratorC    The type of the iterator that provides access to the value in the view `c`
     * @param a             An iterator to the beginning of the view `a`
     * @param b             An iterator to the beginning of the view `b`
     * @param c             An iterator to the beginning of the view `c`
     * @param numElements   The number of elements in the views `a`, `b` and `c`
     */
    template<typename IteratorA, typename IteratorB, typename IteratorC>
    static inline void difference(IteratorA a, IteratorB b, IteratorC c, uint32 numElements) {
        for (uint32 i = 0; i < numElements; i++) {
            a[i] = b[i] - c[i];
        }
    }

    /**
     * Sets all elements in a view `a` to the difference between the elements in two other views `b` and `c`, such that
     * `a = b - c`. The indices of elements in the view `b` that correspond to the elements in the views `a` and `c` are
     * obtained from an additional view.
     *
     * @tparam IteratorA        The type of the iterator that provides access to the value in the view `a`
     * @tparam IteratorB        The type of the iterator that provides access to the value in the view `b`
     * @tparam IteratorC        The type of the iterator that provides access to the value in the view `c`
     * @tparam IndexIterator    The type of the iterator that provides access to the indices
     * @param a                 An iterator to the beginning of the view `a`
     * @param b                 An iterator to the beginning of the view `b`
     * @param c                 An iterator to the beginning of the view `c`
     * @param indices           An iterator to the beginning of the view that provides access to the indices of the
     *                          elements in the view `b` that correspond to the elements in the views `a` and `c`
     * @param numElements       The number of elements in the view `a`
     */
    template<typename IteratorA, typename IteratorB, typename IteratorC, typename IndexIterator>
    static inline void difference(IteratorA a, const IteratorB b, const IteratorC c, IndexIterator indices,
                                  uint32 numElements) {
        for (uint32 i = 0; i < numElements; i++) {
            uint32 index = indices[i];
            a[i] = b[index] - c[i];
        }
    }
}
