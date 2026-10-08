/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"
#include "mlrl/common/util/xsimd.hpp"

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template<typename Arch, typename T>
    uint32 getPadding(Arch, uint32 numElements);

    #if defined(__aarch64__) || defined(_M_ARM64)
    extern template uint32 getPadding<xsimd::neon64, float32>(xsimd::neon64, uint32);
    extern template uint32 getPadding<xsimd::neon64, float64>(xsimd::neon64, uint32);
    extern template uint32 getPadding<xsimd::neon64, uint16>(xsimd::neon64, uint32);
    extern template uint32 getPadding<xsimd::neon64, uint32>(xsimd::neon64, uint32);
    #else
    extern template uint32 getPadding<xsimd::sse2, float32>(xsimd::sse2, uint32);
    extern template uint32 getPadding<xsimd::sse2, float64>(xsimd::sse2, uint32);
    extern template uint32 getPadding<xsimd::sse2, uint16>(xsimd::sse2, uint32);
    extern template uint32 getPadding<xsimd::sse2, uint32>(xsimd::sse2, uint32);
    extern template uint32 getPadding<xsimd::avx, float32>(xsimd::avx, uint32);
    extern template uint32 getPadding<xsimd::avx, float64>(xsimd::avx, uint32);
    extern template uint32 getPadding<xsimd::avx, uint16>(xsimd::avx, uint32);
    extern template uint32 getPadding<xsimd::avx, uint32>(xsimd::avx, uint32);
    extern template uint32 getPadding<xsimd::avx2, float32>(xsimd::avx2, uint32);
    extern template uint32 getPadding<xsimd::avx2, float64>(xsimd::avx2, uint32);
    extern template uint32 getPadding<xsimd::avx2, uint16>(xsimd::avx2, uint32);
    extern template uint32 getPadding<xsimd::avx2, uint32>(xsimd::avx2, uint32);
    extern template uint32 getPadding<xsimd::avx512f, float32>(xsimd::avx512f, uint32);
    extern template uint32 getPadding<xsimd::avx512f, float64>(xsimd::avx512f, uint32);
    extern template uint32 getPadding<xsimd::avx512f, uint16>(xsimd::avx512f, uint32);
    extern template uint32 getPadding<xsimd::avx512f, uint32>(xsimd::avx512f, uint32);
    #endif

    template<typename Arch, typename T>
    T* allocateMemory(Arch, uint32 numElements, bool init);

    #if defined(__aarch64__) || defined(_M_ARM64)
    extern template float32* allocateMemory<xsimd::neon64, float32>(xsimd::neon64, uint32, bool);
    extern template float64* allocateMemory<xsimd::neon64, float64>(xsimd::neon64, uint32, bool);
    extern template uint16* allocateMemory<xsimd::neon64, uint16>(xsimd::neon64, uint32, bool);
    extern template uint32* allocateMemory<xsimd::neon64, uint32>(xsimd::neon64, uint32, bool);
    #else
    extern template float32* allocateMemory<xsimd::sse2, float32>(xsimd::sse2, uint32, bool);
    extern template float64* allocateMemory<xsimd::sse2, float64>(xsimd::sse2, uint32, bool);
    extern template uint16* allocateMemory<xsimd::sse2, uint16>(xsimd::sse2, uint32, bool);
    extern template uint32* allocateMemory<xsimd::sse2, uint32>(xsimd::sse2, uint32, bool);
    extern template float32* allocateMemory<xsimd::avx, float32>(xsimd::avx, uint32, bool);
    extern template float64* allocateMemory<xsimd::avx, float64>(xsimd::avx, uint32, bool);
    extern template uint16* allocateMemory<xsimd::avx, uint16>(xsimd::avx, uint32, bool);
    extern template uint32* allocateMemory<xsimd::avx, uint32>(xsimd::avx, uint32, bool);
    extern template float32* allocateMemory<xsimd::avx2, float32>(xsimd::avx2, uint32, bool);
    extern template float64* allocateMemory<xsimd::avx2, float64>(xsimd::avx2, uint32, bool);
    extern template uint16* allocateMemory<xsimd::avx2, uint16>(xsimd::avx2, uint32, bool);
    extern template uint32* allocateMemory<xsimd::avx2, uint32>(xsimd::avx2, uint32, bool);
    extern template float32* allocateMemory<xsimd::avx512f, float32>(xsimd::avx512f, uint32, bool);
    extern template float64* allocateMemory<xsimd::avx512f, float64>(xsimd::avx512f, uint32, bool);
    extern template uint16* allocateMemory<xsimd::avx512f, uint16>(xsimd::avx512f, uint32, bool);
    extern template uint32* allocateMemory<xsimd::avx512f, uint32>(xsimd::avx512f, uint32, bool);
    #endif

    template<typename Arch, typename T>
    T* reallocateMemory(Arch, T* array, uint32 previousElements, uint32 newElements);

    #if defined(__aarch64__) || defined(_M_ARM64)
    extern template float32* reallocateMemory<xsimd::neon64, float32>(xsimd::neon64, float32*, uint32, uint32);
    extern template float64* reallocateMemory<xsimd::neon64, float64>(xsimd::neon64, float64*, uint32, uint32);
    extern template uint16* reallocateMemory<xsimd::neon64, uint16>(xsimd::neon64, uint16*, uint32, uint32);
    extern template uint32* reallocateMemory<xsimd::neon64, uint32>(xsimd::neon64, uint32*, uint32, uint32);
    #else
    extern template float32* reallocateMemory<xsimd::sse2, float32>(xsimd::sse2, float32*, uint32, uint32);
    extern template float64* reallocateMemory<xsimd::sse2, float64>(xsimd::sse2, float64*, uint32, uint32);
    extern template uint16* reallocateMemory<xsimd::sse2, uint16>(xsimd::sse2, uint16*, uint32, uint32);
    extern template uint32* reallocateMemory<xsimd::sse2, uint32>(xsimd::sse2, uint32*, uint32, uint32);
    extern template float32* reallocateMemory<xsimd::avx, float32>(xsimd::avx, float32*, uint32, uint32);
    extern template float64* reallocateMemory<xsimd::avx, float64>(xsimd::avx, float64*, uint32, uint32);
    extern template uint16* reallocateMemory<xsimd::avx, uint16>(xsimd::avx, uint16*, uint32, uint32);
    extern template uint32* reallocateMemory<xsimd::avx, uint32>(xsimd::avx, uint32*, uint32, uint32);
    extern template float32* reallocateMemory<xsimd::avx2, float32>(xsimd::avx2, float32*, uint32, uint32);
    extern template float64* reallocateMemory<xsimd::avx2, float64>(xsimd::avx2, float64*, uint32, uint32);
    extern template uint16* reallocateMemory<xsimd::avx2, uint16>(xsimd::avx2, uint16*, uint32, uint32);
    extern template uint32* reallocateMemory<xsimd::avx2, uint32>(xsimd::avx2, uint32*, uint32, uint32);
    extern template float32* reallocateMemory<xsimd::avx512f, float32>(xsimd::avx512f, float32*, uint32, uint32);
    extern template float64* reallocateMemory<xsimd::avx512f, float64>(xsimd::avx512f, float64*, uint32, uint32);
    extern template uint16* reallocateMemory<xsimd::avx512f, uint16>(xsimd::avx512f, uint16*, uint32, uint32);
    extern template uint32* reallocateMemory<xsimd::avx512f, uint32>(xsimd::avx512f, uint32*, uint32, uint32);
    #endif

    template<typename Arch, typename T>
    void freeMemory(Arch, T* array);

    #if defined(__aarch64__) || defined(_M_ARM64)
    extern template void freeMemory<xsimd::neon64, float32>(xsimd::neon64, float32*);
    extern template void freeMemory<xsimd::neon64, float64>(xsimd::neon64, float64*);
    extern template void freeMemory<xsimd::neon64, uint16>(xsimd::neon64, uint16*);
    extern template void freeMemory<xsimd::neon64, uint32>(xsimd::neon64, uint32*);
    #else
    extern template void freeMemory<xsimd::sse2, float32>(xsimd::sse2, float32*);
    extern template void freeMemory<xsimd::sse2, float64>(xsimd::sse2, float64*);
    extern template void freeMemory<xsimd::sse2, uint16>(xsimd::sse2, uint16*);
    extern template void freeMemory<xsimd::sse2, uint32>(xsimd::sse2, uint32*);
    extern template void freeMemory<xsimd::avx, float32>(xsimd::avx, float32*);
    extern template void freeMemory<xsimd::avx, float64>(xsimd::avx, float64*);
    extern template void freeMemory<xsimd::avx, uint16>(xsimd::avx, uint16*);
    extern template void freeMemory<xsimd::avx, uint32>(xsimd::avx, uint32*);
    extern template void freeMemory<xsimd::avx2, float32>(xsimd::avx2, float32*);
    extern template void freeMemory<xsimd::avx2, float64>(xsimd::avx2, float64*);
    extern template void freeMemory<xsimd::avx2, uint16>(xsimd::avx2, uint16*);
    extern template void freeMemory<xsimd::avx2, uint32>(xsimd::avx2, uint32*);
    extern template void freeMemory<xsimd::avx512f, float32>(xsimd::avx512f, float32*);
    extern template void freeMemory<xsimd::avx512f, float64>(xsimd::avx512f, float64*);
    extern template void freeMemory<xsimd::avx512f, uint16>(xsimd::avx512f, uint16*);
    extern template void freeMemory<xsimd::avx512f, uint32>(xsimd::avx512f, uint32*);
    #endif
}
#endif
