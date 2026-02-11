/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "config.hpp"
#include "mlrl/common/data/types.hpp"

#include <string>
#include <vector>

#if SIMD_SUPPORT_ENABLED
    #include <xsimd/xsimd.hpp>
#endif

namespace util {

    /**
     * Returns the names of all instruction set extensions for SIMD operations available on the machine.
     *
     * @return An `std::vector` that contains the names of all supported instruction set extensions
     */
    static inline std::vector<std::string> getSupportedSimdExtensions() {
        std::vector<std::string> names;

#if SIMD_SUPPORT_ENABLED
        xsimd::all_architectures::for_each([&](auto architecture) {
            if (xsimd::available_architectures().has(architecture)) {
                names.emplace_back(architecture.name());
            }
        });
#endif

        return names;
    }
}

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
            typedef xsimd::batch<T> batch;
            constexpr std::size_t batchSize = batch::size;
            uint32 batchEnd = numElements - (numElements % batchSize);
            uint32 i = 0;

            for (; i < batchEnd; i += batchSize) {
                batch::load_unaligned(from + i).store_unaligned(to + i);
            }

            for (; i < numElements; i++) {
                to[i] = from[i];
            }
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
            typedef xsimd::batch<T> batch;
            constexpr std::size_t batchSize = batch::size;
            uint32 batchEnd = numElements - (numElements % batchSize);
            uint32 i = 0;

            for (; i < batchEnd; i += batchSize) {
                batch batchA = batch::load_unaligned(a + i);
                batch batchB = batch::load_unaligned(b + i);
                (batchA + batchB).store_unaligned(a + i);
            }

            for (; i < numElements; i++) {
                a[i] += b[i];
            }
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
            typedef xsimd::batch<T> batch;
            constexpr std::size_t batchSize = batch::size;
            uint32 batchEnd = numElements - (numElements % batchSize);
            uint32 i = 0;

            for (; i < batchEnd; i += batchSize) {
                batch batchA = batch::load_unaligned(a + i);
                batch batchB = batch::load_unaligned(b + i);
                (batchA + (batchB * weight)).store_unaligned(a + i);
            }

            for (; i < numElements; i++) {
                a[i] += (b[i] * weight);
            }
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
            typedef xsimd::batch<T> batch;
            constexpr std::size_t batchSize = batch::size;
            uint32 batchEnd = numElements - (numElements % batchSize);
            uint32 i = 0;

            for (; i < batchEnd; i += batchSize) {
                T tmp[batchSize];

                for (std::size_t j = 0; j < batchSize; j++) {
                    uint32 index = indices[i + j];
                    tmp[j] = b[index];
                }

                batch batchA = batch::load_unaligned(a + i);
                batch batchTmp = batch::load_unaligned(tmp);
                (batchA + batchTmp).store_unaligned(a + i);
            }

            for (; i < numElements; i++) {
                uint32 index = indices[i];
                a[i] += b[index];
            }
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
            typedef xsimd::batch<T> batch;
            constexpr std::size_t batchSize = batch::size;
            uint32 batchEnd = numElements - (numElements % batchSize);
            uint32 i = 0;

            for (; i < batchEnd; i += batchSize) {
                T tmp[batchSize];

                for (std::size_t j = 0; j < batchSize; j++) {
                    uint32 index = indices[i + j];
                    tmp[j] = b[index];
                }

                batch batchA = batch::load_unaligned(a + i);
                batch batchTmp = batch::load_unaligned(tmp);
                (batchA + (batchTmp * weight)).store_unaligned(a + i);
            }

            for (; i < numElements; i++) {
                uint32 index = indices[i];
                a[i] += (b[index] * weight);
            }
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
            typedef xsimd::batch<T> batch;
            constexpr std::size_t batchSize = batch::size;
            uint32 batchEnd = numElements - (numElements % batchSize);
            uint32 i = 0;

            for (; i < batchEnd; i += batchSize) {
                batch batchA = batch::load_unaligned(a + i);
                batch batchB = batch::load_unaligned(b + i);
                (batchA - batchB).store_unaligned(a + i);
            }

            for (; i < numElements; i++) {
                a[i] -= b[i];
            }
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
            typedef xsimd::batch<T> batch;
            constexpr std::size_t batchSize = batch::size;
            uint32 batchEnd = numElements - (numElements % batchSize);
            uint32 i = 0;

            for (; i < batchEnd; i += batchSize) {
                batch batchA = batch::load_unaligned(a + i);
                batch batchB = batch::load_unaligned(b + i);
                (batchA - (batchB * weight)).store_unaligned(a + i);
            }

            for (; i < numElements; i++) {
                a[i] -= (b[i] * weight);
            }
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
            typedef xsimd::batch<T> batch;
            constexpr std::size_t batchSize = batch::size;
            uint32 batchEnd = numElements - (numElements % batchSize);
            uint32 i = 0;

            for (; i < batchEnd; i += batchSize) {
                batch batchB = batch::load_unaligned(b + i);
                batch batchC = batch::load_unaligned(c + i);
                (batchB - batchC).store_unaligned(a + i);
            }

            for (; i < numElements; i++) {
                a[i] = b[i] - c[i];
            }
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
            typedef xsimd::batch<T> batch;
            constexpr std::size_t batchSize = batch::size;
            uint32 batchEnd = numElements - (numElements % batchSize);
            uint32 i = 0;

            for (; i < batchEnd; i += batchSize) {
                T tmp[batchSize];

                for (std::size_t j = 0; j < batchSize; j++) {
                    uint32 index = indices[i + j];
                    tmp[j] = b[index];
                }

                batch batchTmp = batch::load_unaligned(tmp);
                batch batchC = batch::load_unaligned(c + i);
                (batchTmp - batchC).store_unaligned(a + i);
            }

            for (; i < numElements; i++) {
                uint32 index = indices[i];
                a[i] = b[index] - c[i];
            }
        }
};
#endif
