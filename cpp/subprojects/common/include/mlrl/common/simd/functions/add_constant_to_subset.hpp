/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"
#include "mlrl/common/util/xsimd.hpp"

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template<typename Arch, typename T>
    void addConstantToSubset(Arch, T* array, T constant, const uint32* indices, uint32 numIndices);

    #if defined(__aarch64__) || defined(_M_ARM64)
    extern template void addConstantToSubset<xsimd::neon64, uint32>(xsimd::neon64, uint32*, uint32, const uint32*,
                                                                    uint32);
    extern template void addConstantToSubset<xsimd::neon64, float32>(xsimd::neon64, float32*, float32, const uint32*,
                                                                     uint32);
    extern template void addConstantToSubset<xsimd::neon64, float64>(xsimd::neon64, float64*, float64, const uint32*,
                                                                     uint32);
    #else
    extern template void addConstantToSubset<xsimd::sse2, uint32>(xsimd::sse2, uint32*, uint32, const uint32*, uint32);
    extern template void addConstantToSubset<xsimd::sse2, float32>(xsimd::sse2, float32*, float32, const uint32*,
                                                                   uint32);
    extern template void addConstantToSubset<xsimd::sse2, float64>(xsimd::sse2, float64*, float64, const uint32*,
                                                                   uint32);
    extern template void addConstantToSubset<xsimd::avx, uint32>(xsimd::avx, uint32*, uint32, const uint32*, uint32);
    extern template void addConstantToSubset<xsimd::avx, float32>(xsimd::avx, float32*, float32, const uint32*, uint32);
    extern template void addConstantToSubset<xsimd::avx, float64>(xsimd::avx, float64*, float64, const uint32*, uint32);
    extern template void addConstantToSubset<xsimd::avx2, uint32>(xsimd::avx2, uint32*, uint32, const uint32*, uint32);
    extern template void addConstantToSubset<xsimd::avx2, float32>(xsimd::avx2, float32*, float32, const uint32*,
                                                                   uint32);
    extern template void addConstantToSubset<xsimd::avx2, float64>(xsimd::avx2, float64*, float64, const uint32*,
                                                                   uint32);
    extern template void addConstantToSubset<xsimd::avx512f, uint32>(xsimd::avx512f, uint32*, uint32, const uint32*,
                                                                     uint32);
    extern template void addConstantToSubset<xsimd::avx512f, float32>(xsimd::avx512f, float32*, float32, const uint32*,
                                                                      uint32);
    extern template void addConstantToSubset<xsimd::avx512f, float64>(xsimd::avx512f, float64*, float64, const uint32*,
                                                                      uint32);
    #endif
}
#endif
