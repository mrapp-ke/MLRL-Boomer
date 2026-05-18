/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/simd/functions/add.hpp"
#include "mlrl/common/simd/functions/add_from_subset.hpp"
#include "mlrl/common/simd/functions/add_weighted.hpp"
#include "mlrl/common/simd/functions/add_weighted_from_subset.hpp"
#include "mlrl/common/simd/functions/copy.hpp"
#include "mlrl/common/simd/functions/difference.hpp"
#include "mlrl/common/simd/functions/difference_with_subset.hpp"
#include "mlrl/common/simd/functions/subtract.hpp"
#include "mlrl/common/simd/functions/subtract_weighted.hpp"

#if SIMD_SUPPORT_ENABLED
/**
 * Implements basic operations for calculating with numerical arrays using single instruction, multiple data (SIMD)
 * operations.
 */
struct SimdVectorMath {
    public:

        /**
         * Copy all elements from one array to another.
         *
         * @tparam T            The type of the elements
         * @param from          A pointer to the beginning of the array to copy from
         * @param to            A pointer to the beginning of the array to copy to
         * @param numElements   The number of elements to be copied
         */
        template<typename T>
        static inline void copy(const T* from, T* to, uint32 numElements) {
            auto dispatched = xsimd::dispatch<util::simd_architectures>([&](auto arch) {
                simd::copy(arch, from, to, numElements);
            });
            dispatched();
        }

        /**
         * Adds the elements in an array `b` to the elements in another array `a`, such that `a = a + b`.
         *
         * @tparam T            The type of the values in the array `a` and `b`
         * @param a             A pointer to the beginning of the array `a`
         * @param b             A pointer to the beginning of the array `b`
         * @param numElements   The number of elements in the arrays `a` and `b`
         */
        template<typename T>
        static inline void add(T* a, const T* b, uint32 numElements) {
            auto dispatched = xsimd::dispatch<util::simd_architectures>([&](auto arch) {
                simd::add(arch, a, b, numElements);
            });
            dispatched();
        }

        /**
         * Adds the elements in an array `b` to the elements in another array `a`. The elements in the array `b` are
         * multiplied by a given weight, such that `a = a + (b * weight)`.
         *
         * @tparam T            The type of the values in the arrays `a` and `b`
         * @tparam Weight       The type of the weight
         * @param a             A pointer to the beginning of the array `a`
         * @param b             A pointer to the beginning of the array `b`
         * @param numElements   The number of elements in the arrays `a` and `b`
         * @param weight        The weight, the elements in the array `b` should be multiplied by
         */
        template<typename T, typename Weight>
        static inline void addWeighted(T* a, const T* b, uint32 numElements, Weight weight) {
            auto dispatched = xsimd::dispatch<util::simd_architectures>([&](auto arch) {
                simd::addWeighted(arch, a, b, numElements, weight);
            });
            dispatched();
        }

        /**
         * Adds the elements in an array `b` to the elements in another array `a`, such that `a = a + b`. The indices of
         * the elements in the array `b` that correspond to the elements in the array `a` are given as an additional
         * array.
         *
         * @tparam T            The type of the values in the arrays `a` and `b`
         * @param a             A pointer to the beginning of the array `a`
         * @param b             A pointer to the beginning of the array `b`
         * @param numElements   The number of elements in the arrays `a` and `b`
         * @param indices       A pointer to the beginning of an array that stores the indices of the elements in the
         *                      array `b` that correspond to the elements in the array `a`
         */
        template<typename T>
        static inline void add(T* a, const T* b, const uint32* indices, uint32 numElements) {
            auto dispatched = xsimd::dispatch<util::simd_architectures>([&](auto arch) {
                simd::addFromSubset(arch, a, b, indices, numElements);
            });
            dispatched();
        }

        /**
         * Adds the elements in an array `b` to the elements in another array `a`. The elements in the array `b` are
         * multiplied by a given weight, such that `a = a + (b * weight)`. The indices of the elements in the array `b`
         * that correspond to the elements in the array `a` are given as an additional array.
         *
         * @tparam T            The type of the values in the arrays `a` and `b`
         * @tparam Weight       The type of the weight
         * @param a             A pointer to the beginning of the array `a`
         * @param b             A pointer to the beginning of the array `b`
         * @param numElements   The number of elements in the arrays `a` and `b`
         * @param weight        The weight, the elements in the array `b` should be multiplied by
         * @param indices       A pointer to the beginning of a array that stores the indices of the elements in the
         *                      array `b` that correspond to the elements in the array `a`
         */
        template<typename T, typename Weight>
        static inline void addWeighted(T* a, const T* b, const uint32* indices, uint32 numElements, Weight weight) {
            auto dispatched = xsimd::dispatch<util::simd_architectures>([&](auto arch) {
                simd::addWeightedFromSubset(arch, a, b, indices, numElements, weight);
            });
            dispatched();
        }

        /**
         * Removes the elements in an array `b` from the elements in another array `a`, such that `a = a - b`.
         *
         * @tparam T            The type of the values in the arrays `a` and `b`
         * @param a             A pointer to the beginning of the array `a`
         * @param b             A pointer to the beginning of the array `b`
         * @param numElements   The number of elements in the arrays `a` and `b`
         */
        template<typename T>
        static inline void subtract(T* a, const T* b, uint32 numElements) {
            auto dispatched = xsimd::dispatch<util::simd_architectures>([&](auto arch) {
                simd::subtract(arch, a, b, numElements);
            });
            dispatched();
        }

        /**
         * Removes the elements in an array `b` from the elements in another array `a`. The elements in the array `b`
         * are multiplied by a given weight, such that `a = a - (b * weight)`.
         *
         * @tparam T            The type of the values in the arrays `a` and `b`
         * @tparam Weight       The type of the weight
         * @param a             A pointer to the beginning of the array `a`
         * @param b             A pointer to the beginning of the array `b`
         * @param numElements   The number of elements in the arrays `a` and `b`
         * @param weight        The weight, the elements in the array `b` should be multiplied by
         */
        template<typename T, typename Weight>
        static inline void subtractWeighted(T* a, const T* b, uint32 numElements, Weight weight) {
            auto dispatched = xsimd::dispatch<util::simd_architectures>([&](auto arch) {
                simd::subtractWeighted(arch, a, b, numElements, weight);
            });
            dispatched();
        }

        /**
         * Sets all elements in an array `a` to the difference between the elements in two other arrays `b` and `c`,
         * such that `a = b - c`.
         *
         * @tparam T            The type of the values in the arrays `a`, `b` and `c`
         * @param a             A pointer to the beginning of the array `a`
         * @param b             A pointer to the beginning of the array `b`
         * @param c             A pointer to the beginning of the array `c`
         * @param numElements   The number of elements in the arrays `a`, `b` and `c`
         */
        template<typename T>
        static inline void difference(T* a, const T* b, const T* c, uint32 numElements) {
            auto dispatched = xsimd::dispatch<util::simd_architectures>([&](auto arch) {
                simd::difference(arch, a, b, c, numElements);
            });
            dispatched();
        }

        /**
         * Sets all elements in an array `a` to the difference between the elements in two other arrays `b` and `c`,
         * such that `a = b - c`. The indices of elements in the array `b` that correspond to the elements in the arrays
         * `a` and `c` are obtained from an additional array.
         *
         * @tparam T            The type of the values in the arrays `a`, `b` and `c`
         * @param a             A pointer to the beginning of the array `a`
         * @param b             A pointer to the beginning of the array `b`
         * @param c             A pointer to the beginning of the array `c`
         * @param indices       A pointer to the beginning of the array that provides access to the indices of the
         *                      elements in the array `b` that correspond to the elements in the arrays `a` and `c`
         * @param numElements   The number of elements in the array `a`
         */
        template<typename T>
        static inline void difference(T* a, const T* b, const T* c, const uint32* indices, uint32 numElements) {
            auto dispatched = xsimd::dispatch<util::simd_architectures>([&](auto arch) {
                simd::differenceWithSubset(arch, a, b, c, indices, numElements);
            });
            dispatched();
        }
};
#endif
