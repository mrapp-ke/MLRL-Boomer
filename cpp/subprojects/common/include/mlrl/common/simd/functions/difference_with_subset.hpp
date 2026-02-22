/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"
#include "mlrl/common/util/xsimd.hpp"

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template<typename Arch, typename T>
    void differenceWithSubset(Arch, T* a, const T* b, const T* c, const uint32* indices, uint32 numElements);

    #if defined(__aarch64__) || defined(_M_ARM64)
    extern template void differenceWithSubset<xsimd::neon64, float32>(xsimd::neon64, float32*, const float32*,
                                                                      const float32*, const uint32*, uint32);
    extern template void differenceWithSubset<xsimd::neon64, float64>(xsimd::neon64, float64*, const float64*,
                                                                      const float64*, const uint32*, uint32);
    #else
    extern template void differenceWithSubset<xsimd::sse2, float32>(xsimd::sse2, float32*, const float32*,
                                                                    const float32*, const uint32*, uint32);
    extern template void differenceWithSubset<xsimd::sse2, float64>(xsimd::sse2, float64*, const float64*,
                                                                    const float64*, const uint32*, uint32);
    extern template void differenceWithSubset<xsimd::avx, float32>(xsimd::avx, float32*, const float32*, const float32*,
                                                                   const uint32*, uint32);
    extern template void differenceWithSubset<xsimd::avx, float64>(xsimd::avx, float64*, const float64*, const float64*,
                                                                   const uint32*, uint32);
    extern template void differenceWithSubset<xsimd::avx2, float32>(xsimd::avx2, float32*, const float32*,
                                                                    const float32*, const uint32*, uint32);
    extern template void differenceWithSubset<xsimd::avx2, float64>(xsimd::avx2, float64*, const float64*,
                                                                    const float64*, const uint32*, uint32);
    extern template void differenceWithSubset<xsimd::avx512f, float32>(xsimd::avx512f, float32*, const float32*,
                                                                       const float32*, const uint32*, uint32);
    extern template void differenceWithSubset<xsimd::avx512f, float64>(xsimd::avx512f, float64*, const float64*,
                                                                       const float64*, const uint32*, uint32);
    #endif
}
#endif
