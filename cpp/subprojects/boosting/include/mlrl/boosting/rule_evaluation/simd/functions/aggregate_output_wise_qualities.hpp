/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"
#include "mlrl/common/util/xsimd.hpp"

#if SIMD_SUPPORT_ENABLED
namespace boosting {
    namespace simd {
        template<typename Arch, typename StatisticType>
        StatisticType aggregateOutputWiseQualities(Arch, const StatisticType* scores, const StatisticType* gradients,
                                                   const StatisticType* hessians, uint32 numElements,
                                                   float32 l1RegularizationWeight, float32 l2RegularizationWeight);

    #if defined(__aarch64__) || defined(_M_ARM64)
        extern template float32 aggregateOutputWiseQualities<xsimd::neon64, float32>(xsimd::neon64, const float32*,
                                                                                     const float32*, const float32*,
                                                                                     uint32, float32, float32);
        extern template float64 aggregateOutputWiseQualities<xsimd::neon64, float64>(xsimd::neon64, const float64*,
                                                                                     const float64*, const float64*,
                                                                                     uint32, float32, float32);
    #else
        extern template float32 aggregateOutputWiseQualities<xsimd::sse2, float32>(xsimd::sse2, const float32*,
                                                                                   const float32*, const float32*,
                                                                                   uint32, float32, float32);
        extern template float64 aggregateOutputWiseQualities<xsimd::sse2, float64>(xsimd::sse2, const float64*,
                                                                                   const float64*, const float64*,
                                                                                   uint32, float32, float32);
        extern template float32 aggregateOutputWiseQualities<xsimd::avx, float32>(xsimd::avx, const float32*,
                                                                                  const float32*, const float32*,
                                                                                  uint32, float32, float32);
        extern template float64 aggregateOutputWiseQualities<xsimd::avx, float64>(xsimd::avx, const float64*,
                                                                                  const float64*, const float64*,
                                                                                  uint32, float32, float32);
        extern template float32 aggregateOutputWiseQualities<xsimd::avx2, float32>(xsimd::avx2, const float32*,
                                                                                   const float32*, const float32*,
                                                                                   uint32, float32, float32);
        extern template float64 aggregateOutputWiseQualities<xsimd::avx2, float64>(xsimd::avx2, const float64*,
                                                                                   const float64*, const float64*,
                                                                                   uint32, float32, float32);
        extern template float32 aggregateOutputWiseQualities<xsimd::avx512f, float32>(xsimd::avx512f, const float32*,
                                                                                      const float32*, const float32*,
                                                                                      uint32, float32, float32);
        extern template float64 aggregateOutputWiseQualities<xsimd::avx512f, float64>(xsimd::avx512f, const float64*,
                                                                                      const float64*, const float64*,
                                                                                      uint32, float32, float32);
    #endif

        template<typename Arch, typename StatisticType>
        StatisticType aggregateOutputWiseQualitiesWeighted(Arch, const StatisticType* scores,
                                                           const StatisticType* gradients,
                                                           const StatisticType* hessians, const uint32* weights,
                                                           uint32 numElements, float32 l1RegularizationWeight,
                                                           float32 l2RegularizationWeight);

    #if defined(__aarch64__) || defined(_M_ARM64)
        extern template float32 aggregateOutputWiseQualitiesWeighted<xsimd::neon64, float32>(
          xsimd::neon64, const float32*, const float32*, const float32*, const uint32*, uint32, float32, float32);
        extern template float64 aggregateOutputWiseQualitiesWeighted<xsimd::neon64, float64>(
          xsimd::neon64, const float64*, const float64*, const float64*, const uint32*, uint32, float32, float32);
    #else
        extern template float32 aggregateOutputWiseQualitiesWeighted<xsimd::sse2, float32>(
          xsimd::sse2, const float32*, const float32*, const float32*, const uint32*, uint32, float32, float32);
        extern template float64 aggregateOutputWiseQualitiesWeighted<xsimd::sse2, float64>(
          xsimd::sse2, const float64*, const float64*, const float64*, const uint32*, uint32, float32, float32);
        extern template float32 aggregateOutputWiseQualitiesWeighted<xsimd::avx, float32>(xsimd::avx, const float32*,
                                                                                          const float32*,
                                                                                          const float32*, const uint32*,
                                                                                          uint32, float32, float32);
        extern template float64 aggregateOutputWiseQualitiesWeighted<xsimd::avx, float64>(xsimd::avx, const float64*,
                                                                                          const float64*,
                                                                                          const float64*, const uint32*,
                                                                                          uint32, float32, float32);
        extern template float32 aggregateOutputWiseQualitiesWeighted<xsimd::avx2, float32>(
          xsimd::avx2, const float32*, const float32*, const float32*, const uint32*, uint32, float32, float32);
        extern template float64 aggregateOutputWiseQualitiesWeighted<xsimd::avx2, float64>(
          xsimd::avx2, const float64*, const float64*, const float64*, const uint32*, uint32, float32, float32);
        extern template float32 aggregateOutputWiseQualitiesWeighted<xsimd::avx512f, float32>(
          xsimd::avx512f, const float32*, const float32*, const float32*, const uint32*, uint32, float32, float32);
        extern template float64 aggregateOutputWiseQualitiesWeighted<xsimd::avx512f, float64>(
          xsimd::avx512f, const float64*, const float64*, const float64*, const uint32*, uint32, float32, float32);
    #endif
    }
}
#endif
