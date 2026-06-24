/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/simd/functions/memory.hpp"

#if SIMD_SUPPORT_ENABLED
    #include <algorithm>
    #include <cstring>
    #include <new>
    #include <type_traits>
#endif

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template<typename Arch, typename T>
    uint32 getPadding(Arch, uint32 numElements) {
        constexpr std::size_t alignment = Arch::alignment();
        constexpr uint32 numElementsPerBatch = static_cast<uint32>(alignment / sizeof(T));
        uint32 remainder = numElements % numElementsPerBatch;
        return remainder == 0 ? 0 : numElementsPerBatch - remainder;
    }

    template<typename Arch, typename T>
    T* allocateMemory(Arch, uint32 numElements, bool init) {
        constexpr std::size_t alignment = Arch::alignment();
        std::size_t bytes = static_cast<std::size_t>(numElements) * sizeof(T);
        T* ptr = static_cast<T*>(::operator new[](bytes, std::align_val_t {alignment}));

        if (init) {
            if constexpr (std::is_trivial_v<T>) {
                std::memset(ptr, 0, bytes);
            } else {
                std::fill(ptr, &ptr[numElements], 0);
            }
        }

        return ptr;
    }

    template<typename Arch, typename T>
    T* reallocateMemory(Arch arch, T* array, uint32 previousElements, uint32 newElements) {
        constexpr std::size_t alignment = Arch::alignment();
        T* ptr = simd::allocateMemory<Arch, T>(arch, newElements, false);
        std::memcpy(ptr, array, std::min(previousElements, newElements) * sizeof(T));
        ::operator delete[](array, std::align_val_t {alignment});
        return ptr;
    }

    template<typename Arch, typename T>
    void freeMemory(Arch, T* array) {
        ::operator delete[](array, std::align_val_t {Arch::alignment()});
    }
}
#endif
