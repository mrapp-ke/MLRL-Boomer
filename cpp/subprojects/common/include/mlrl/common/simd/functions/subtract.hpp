/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"
#include "mlrl/common/util/xsimd.hpp"

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template<typename Arch, typename T>
    void subtract(Arch, T* a, const T* b, uint32 numElements) {
        typedef xsimd::batch<T, Arch> batch;
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

    #if defined(__aarch64__) || defined(_M_ARM64)
    extern template void subtract<xsimd::neon64, float32>(xsimd::neon64, float32*, const float32*, uint32);
    extern template void subtract<xsimd::neon64, float64>(xsimd::neon64, float64*, const float64*, uint32);
    #else
    extern template void subtract<xsimd::sse2, float32>(xsimd::sse2, float32*, const float32*, uint32);
    extern template void subtract<xsimd::sse2, float64>(xsimd::sse2, float64*, const float64*, uint32);
    extern template void subtract<xsimd::avx, float32>(xsimd::avx, float32*, const float32*, uint32);
    extern template void subtract<xsimd::avx, float64>(xsimd::avx, float64*, const float64*, uint32);
    extern template void subtract<xsimd::avx2, float32>(xsimd::avx2, float32*, const float32*, uint32);
    extern template void subtract<xsimd::avx2, float64>(xsimd::avx2, float64*, const float64*, uint32);
    extern template void subtract<xsimd::avx512f, float32>(xsimd::avx512f, float32*, const float32*, uint32);
    extern template void subtract<xsimd::avx512f, float64>(xsimd::avx512f, float64*, const float64*, uint32);
    #endif
}
#endif
