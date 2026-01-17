/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"

namespace util {

    /**
     * Copy all elements from one view to another.
     *
     * @tparam T            The type of the elements
     * @param from          A pointer to the beginning of the view to copy from
     * @param to            A pointer to the beginning of the view to copy to
     * @param numElements   The number of elements to be copied
     */
    template<typename T>
    static inline void copy(const T* from, T* to, uint32 numElements) {
        for (uint32 i = 0; i < numElements; i++) {
            to[i] = from[i];
        }
    }

    /**
     * Adds the elements in a view `b` to the elements in another view `a`, such that `a = a + b`.
     *
     * @tparam T            The type of the values in the views `a` and `b`
     * @param a             A pointer to the beginning of the view `a`
     * @param b             A pointer to the beginning of the view `b`
     * @param numElements   The number of elements in the views `a` and `b`
     */
    template<typename T>
    static inline void add(T* a, const T* b, uint32 numElements) {
        for (uint32 i = 0; i < numElements; i++) {
            a[i] += b[i];
        }
    }

    /**
     * Adds the elements in a view `b` to the elements in another view `a`. The elements in the view `b` are multiplied
     * by a given weight, such that `a = a + (b * weight)`.
     *
     * @tparam T            The type of the values in the views `a` and `b`
     * @tparam Weight       The type of the weight
     * @param a             A pointer to the beginning of the view `a`
     * @param b             A pointer to the beginning of the view `b`
     * @param numElements   The number of elements in the views `a` and `b`
     * @param weight        The weight, the elements in the view `b` should be multiplied by
     */
    template<typename T, typename Weight>
    static inline void addWeighted(T* a, const T* b, uint32 numElements, Weight weight) {
        for (uint32 i = 0; i < numElements; i++) {
            a[i] += (b[i] * weight);
        }
    }

    /**
     * Adds the elements in a view `b` to the elements in another view `a`, such that `a = a + b`. The indices of the
     * elements in the view `b` that correspond to the elements in the view `a` are given as an additional view.
     *
     * @tparam T            The type of the values in the views `a` and `b`
     * @param a             A pointer to the beginning of the view `a`
     * @param b             A pointer to the beginning of the view `b`
     * @param numElements   The number of elements in the views `a` and `b`
     * @param indices       A pointer to the beginning of a view that stores the indices of the elements in the view `b`
     *                      that correspond to the elements in the view `a`
     */
    template<typename T>
    static inline void add(T* a, const T* b, const uint32* indices, uint32 numElements) {
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
     * @tparam T            The type of the values in the views `a` and `b`
     * @tparam Weight       The type of the weight
     * @param a             A pointer to the beginning of the view `a`
     * @param b             A pointer to the beginning of the view `b`
     * @param numElements   The number of elements in the views `a` and `b`
     * @param weight        The weight, the elements in the view `b` should be multiplied by
     * @param indices       A pointer to the beginning of a view that stores the indices of the elements in the view `b`
     *                      that correspond to the elements in the view `a`
     */
    template<typename T, typename Weight>
    static inline void addWeighted(T* a, const T* b, const uint32* indices, uint32 numElements, Weight weight) {
        for (uint32 i = 0; i < numElements; i++) {
            uint32 index = indices[i];
            a[i] += (b[index] * weight);
        }
    }

    /**
     * Removes the elements in a view `b` from the elements in another view `a`, such that `a = a - b`.
     *
     * @tparam T            The type of the values in the views `a` and `b`
     * @param a             A pointer to the beginning of the view `a`
     * @param b             A pointer to the beginning of the view `b`
     * @param numElements   The number of elements in the views `a` and `b`
     */
    template<typename T>
    static inline void subtract(T* a, const T* b, uint32 numElements) {
        for (uint32 i = 0; i < numElements; i++) {
            a[i] -= b[i];
        }
    }

    /**
     * Removes the elements in a view `b` from the elements in another view `a`. The elements in the view `b` are
     * multiplied by a given weight, such that `a = a - (b * weight)`.
     *
     * @tparam T            The type of the values in the views `a` and `b`
     * @tparam Weight       The type of the weight
     * @param a             A pointer to the beginning of the view `a`
     * @param b             A pointer to the beginning of the view `b`
     * @param numElements   The number of elements in the views `a` and `b`
     * @param weight        The weight, the elements in the view `b` should be multiplied by
     */
    template<typename T, typename Weight>
    static inline void subtractWeighted(T* a, const T* b, uint32 numElements, Weight weight) {
        for (uint32 i = 0; i < numElements; i++) {
            a[i] -= (b[i] * weight);
        }
    }

    /**
     * Sets all elements in a view `a` to the difference between the elements in two other views `b` and `c`, such that
     * `a = b - c`.
     *
     * @tparam T            The type of the values in the views `a`, `b` and `c`
     * @param a             A pointer to the beginning of the view `a`
     * @param b             A pointer to the beginning of the view `b`
     * @param c             A pointer to the beginning of the view `c`
     * @param numElements   The number of elements in the views `a`, `b` and `c`
     */
    template<typename T>
    static inline void difference(T* a, const T* b, const T* c, uint32 numElements) {
        for (uint32 i = 0; i < numElements; i++) {
            a[i] = b[i] - c[i];
        }
    }

    /**
     * Sets all elements in a view `a` to the difference between the elements in two other views `b` and `c`, such that
     * `a = b - c`. The indices of elements in the view `b` that correspond to the elements in the views `a` and `c` are
     * obtained from an additional view.
     *
     * @tparam T            The type of the values in the views `a`, `b` and `c`
     * @param a             A pointer to the beginning of the view `a`
     * @param b             A pointer to the beginning of the view `b`
     * @param c             A pointer to the beginning of the view `c`
     * @param indices       A pointer to the beginning of the view that provides access to the indices of the elements
     *                      in the view `b` that correspond to the elements in the views `a` and `c`
     * @param numElements   The number of elements in the view `a`
     */
    template<typename T>
    static inline void difference(T* a, const T* b, const T* c, const uint32* indices, uint32 numElements) {
        for (uint32 i = 0; i < numElements; i++) {
            uint32 index = indices[i];
            a[i] = b[index] - c[i];
        }
    }
}
