/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"

#include <algorithm>
#include <cstddef>

namespace util {

    /**
     * Sets all elements in a view to zero.
     *
     * @tparam Iterator     The type of the iterator that provides access to the values in the view
     * @param iterator      An iterator to the beginning of the view
     * @param numElements   The number of elements in the view
     */
    template<typename Iterator>
    static inline void setViewToZeros(Iterator iterator, uint32 numElements) {
        std::fill(iterator, iterator + numElements, 0);
    }

    /**
     * Sets all elements in a view to a specific value.
     *
     * @tparam Iterator     The type of the iterator that provides access to the values in the view
     * @tparam Value        The type of the value to be set
     * @param iterator      An iterator to to the beginning of the view
     * @param numElements   The number of elements in the view
     * @param value         The value to be set
     */
    template<typename Iterator, typename T>
    static inline void setViewToValue(Iterator iterator, uint32 numElements, T value) {
        std::fill(iterator, iterator + numElements, value);
    }

    /**
     * Sets the elements in a view to increasing values.
     *
     * @tparam Iterator     The type of the iterator that provides access to the values in the view
     * @tparam Value        The type of the values to be set
     * @param iterator      An iterator to the beginning of the view
     * @param numElements   The number of elements in the view
     * @param start         The value to start at
     * @param increment     The difference between the values
     */
    template<typename Iterator, typename Value>
    static inline void setViewToIncreasingValues(Iterator iterator, uint32 numElements, Value start, Value increment) {
        Value nextValue = start;

        for (uint32 i = 0; i < numElements; i++) {
            iterator[i] = nextValue;
            nextValue += increment;
        }
    }

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
    static inline void copyView(FromIterator from, ToIterator to, uint32 numElements) {
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
     */
    template<typename IteratorA, typename IteratorB, typename Weight>
    static inline void addToViewWeighted(IteratorA a, IteratorB b, uint32 numElements, Weight weight) {
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
     */
    template<typename IteratorA, typename IteratorB, typename IndexIterator, typename Weight>
    static inline void addToViewWeighted(IteratorA a, IteratorB b, IndexIterator indices, uint32 numElements,
                                         Weight weight) {
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
     */
    template<typename IteratorA, typename IteratorB, typename Weight>
    static inline void removeFromView(IteratorA a, IteratorB b, uint32 numElements, Weight weight) {
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
    static inline void setViewToDifference(IteratorA a, IteratorB b, IteratorC c, uint32 numElements) {
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
     * elements in the view `b` that correspond to the elements in the views `a` and `c`
     * @param numElements       The number of elements in the view `a`
     */
    template<typename IteratorA, typename IteratorB, typename IteratorC, typename IndexIterator>
    static inline void setViewToDifference(IteratorA a, const IteratorB b, const IteratorC c, IndexIterator indices,
                                           uint32 numElements) {
        for (uint32 i = 0; i < numElements; i++) {
            uint32 index = indices[i];
            a[i] = b[index] - c[i];
        }
    }

    /**
     * Calculates and returns a hash value from a view.
     *
     * @tparam Iterator     The type of the iterator that provides access to the values in the view
     * @param iterator      An iterator to the beginning of the view
     * @param numElements   The number of elements in the view
     * @return              The hash value that has been calculated
     */
    template<typename Iterator>
    static inline constexpr std::size_t hashView(Iterator iterator, uint32 numElements) {
        std::size_t hashValue = (std::size_t) numElements;

        for (uint32 i = 0; i < numElements; i++) {
            hashValue ^= iterator[i] + 0x9e3779b9 + (hashValue << 6) + (hashValue >> 2);
        }

        return hashValue;
    }

    /**
     * Returns whether all elements of two views are equal or not.
     *
     * @tparam FirstIterator    The type of the iterator that provides access to the values in the first view
     * @tparam SecondIterator   The type of the iterator that provides access to the values in the second view
     * @param first             An iterator to the beginning of the first view
     * @param numFirst          The number of elements in the first view
     * @param second            An iterator to the beginning of the second view
     * @param numSecond         The number of elements in the second view
     * @return                  True, if all elements of both views are equal, false otherwise
     */
    template<typename FirstIterator, typename SecondIterator>
    static inline constexpr bool compareViews(FirstIterator first, uint32 numFirst, SecondIterator second,
                                              uint32 numSecond) {
        if (numFirst != numSecond) {
            return false;
        }

        for (uint32 i = 0; i < numFirst; i++) {
            if (!isEqual(first[i], second[i])) {
                return false;
            }
        }

        return true;
    }

}
