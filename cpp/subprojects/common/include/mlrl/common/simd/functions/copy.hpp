/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"
#include "mlrl/common/util/xsimd.hpp"

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template<typename Arch, typename T>
    void copy(Arch, const T* from, T* to, uint32 numElements);

    #if defined(__aarch64__) || defined(_M_ARM64)
    extern template void copy<xsimd::neon64, float32>(xsimd::neon64, const float32*, float32*, uint32);
    extern template void copy<xsimd::neon64, float64>(xsimd::neon64, const float64*, float64*, uint32);
    #else
    extern template void copy<xsimd::sse2, float32>(xsimd::sse2, const float32*, float32*, uint32);
    extern template void copy<xsimd::sse2, float64>(xsimd::sse2, const float64*, float64*, uint32);
    extern template void copy<xsimd::avx, float32>(xsimd::avx, const float32*, float32*, uint32);
    extern template void copy<xsimd::avx, float64>(xsimd::avx, const float64*, float64*, uint32);
    extern template void copy<xsimd::avx2, float32>(xsimd::avx2, const float32*, float32*, uint32);
    extern template void copy<xsimd::avx2, float64>(xsimd::avx2, const float64*, float64*, uint32);
    extern template void copy<xsimd::avx512f, float32>(xsimd::avx512f, const float32*, float32*, uint32);
    extern template void copy<xsimd::avx512f, float64>(xsimd::avx512f, const float64*, float64*, uint32);
    #endif
}
#endif
