/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/simd/functions/memory.hpp"
#include "mlrl/common/util/memory.hpp"

#if SIMD_SUPPORT_ENABLED
/**
 * Provides functions for allocating memory, such that it is properly aligned for being loaded into SIMD registers.
 */
struct SimdMemoryAllocator final {
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
            auto dispatched = xsimd::dispatch<util::simd_architectures>([&](auto arch) {
                return simd::allocateMemory<decltype(arch), T>(arch, numElements, init);
            });
            return dispatched();
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
            auto dispatched = xsimd::dispatch<util::simd_architectures>([&](auto arch) {
                return simd::reallocateMemory<decltype(arch), T>(arch, array, previousElements, newElements);
            });
            return dispatched();
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
                auto dispatched = xsimd::dispatch<util::simd_architectures>([&](auto arch) {
                    return simd::freeMemory<decltype(arch), T>(arch, array);
                });
                return dispatched();
            }
        }
};
#endif
