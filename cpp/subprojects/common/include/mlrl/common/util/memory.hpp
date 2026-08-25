/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"

#include <cstdlib>

/**
 * Provides functions for allocating memory.
 */
struct DefaultMemoryAllocator final {
    public:

        /**
         * Allocates memory to be used by an array of a specific size.
         *
         * @tparam T            The type of the values stored in the array
         * @param numElements   The number of elements in the array
         * @param init          True, if all elements in the array should be default-initialized, false otherwise
         * @return              A pointer to the allocated memory
         */
        template<typename T>
        static inline constexpr T* allocateMemory(uint32 numElements, bool init = false) {
            if (init) {
                return reinterpret_cast<T*>(calloc(numElements, sizeof(T)));
            } else {
                return reinterpret_cast<T*>(malloc(numElements * sizeof(T)));
            }
        }

        /**
         * Reallocates the memory used by an array in order to resize it.
         *
         * @tparam T                The type of the values stored in the array
         * @param array             A pointer to an array of template type `T`
         * @param previousElements  The number of elements in the original array
         * @param newElements       The number of elements in the resized array
         * @return                  A pointer to the reallocated memory
         */
        template<typename T>
        static inline constexpr T* reallocateMemory(T* array, uint32 previousElements, uint32 newElements) {
            return reinterpret_cast<T*>(realloc(array, newElements * sizeof(T)));
        }

        /**
         * Frees the memory used by an array.
         *
         * @tparam T    The type of the values stored in the array
         * @param array A pointer to an array of template type `T`
         */
        template<typename T>
        static inline constexpr void freeMemory(T* array) {
            if (array) {
                free(array);
            }
        }
};
