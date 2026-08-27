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
        void calculateOutputWiseScores(Arch, const StatisticType* gradients, const StatisticType* hessians,
                                       StatisticType* outputs, uint32 numElements, float32 l1RegularizationWeight,
                                       float32 l2RegularizationWeight);

    #if defined(__aarch64__) || defined(_M_ARM64)
        extern template void calculateOutputWiseScores<xsimd::neon64, float32>(xsimd::neon64, const float32*,
                                                                               const float32*, float32*, uint32,
                                                                               float32, float32);
        extern template void calculateOutputWiseScores<xsimd::neon64, float64>(xsimd::neon64, const float64*,
                                                                               const float64*, float64*, uint32,
                                                                               float32, float32);
    #else
        extern template void calculateOutputWiseScores<xsimd::sse2, float32>(xsimd::sse2, const float32*,
                                                                             const float32*, float32*, uint32, float32,
                                                                             float32);
        extern template void calculateOutputWiseScores<xsimd::sse2, float64>(xsimd::sse2, const float64*,
                                                                             const float64*, float64*, uint32, float32,
                                                                             float32);
        extern template void calculateOutputWiseScores<xsimd::avx, float32>(xsimd::avx, const float32*, const float32*,
                                                                            float32*, uint32, float32, float32);
        extern template void calculateOutputWiseScores<xsimd::avx, float64>(xsimd::avx, const float64*, const float64*,
                                                                            float64*, uint32, float32, float32);
        extern template void calculateOutputWiseScores<xsimd::avx2, float32>(xsimd::avx2, const float32*,
                                                                             const float32*, float32*, uint32, float32,
                                                                             float32);
        extern template void calculateOutputWiseScores<xsimd::avx2, float64>(xsimd::avx2, const float64*,
                                                                             const float64*, float64*, uint32, float32,
                                                                             float32);
        extern template void calculateOutputWiseScores<xsimd::avx512f, float32>(xsimd::avx512f, const float32*,
                                                                                const float32*, float32*, uint32,
                                                                                float32, float32);
        extern template void calculateOutputWiseScores<xsimd::avx512f, float64>(xsimd::avx512f, const float64*,
                                                                                const float64*, float64*, uint32,
                                                                                float32, float32);
    #endif

        template<typename Arch, typename StatisticType>
        void calculateOutputWiseScoresWeighted(Arch, const StatisticType* gradients, const StatisticType* hessians,
                                               const uint32* weights, StatisticType* outputs, uint32 numElements,
                                               float32 l1RegularizationWeight, float32 l2RegularizationWeight);

    #if defined(__aarch64__) || defined(_M_ARM64)
        extern template void calculateOutputWiseScoresWeighted<xsimd::neon64, float32>(xsimd::neon64, const float32*,
                                                                                       const float32*, const uint32*,
                                                                                       float32*, uint32, float32,
                                                                                       float32);
        extern template void calculateOutputWiseScoresWeighted<xsimd::neon64, float64>(xsimd::neon64, const float64*,
                                                                                       const float64*, const uint32*,
                                                                                       float64*, uint32, float32,
                                                                                       float32);
    #else
        extern template void calculateOutputWiseScoresWeighted<xsimd::sse2, float32>(xsimd::sse2, const float32*,
                                                                                     const float32*, const uint32*,
                                                                                     float32*, uint32, float32,
                                                                                     float32);
        extern template void calculateOutputWiseScoresWeighted<xsimd::sse2, float64>(xsimd::sse2, const float64*,
                                                                                     const float64*, const uint32*,
                                                                                     float64*, uint32, float32,
                                                                                     float32);
        extern template void calculateOutputWiseScoresWeighted<xsimd::avx, float32>(xsimd::avx, const float32*,
                                                                                    const float32*, const uint32*,
                                                                                    float32*, uint32, float32, float32);
        extern template void calculateOutputWiseScoresWeighted<xsimd::avx, float64>(xsimd::avx, const float64*,
                                                                                    const float64*, const uint32*,
                                                                                    float64*, uint32, float32, float32);
        extern template void calculateOutputWiseScoresWeighted<xsimd::avx2, float32>(xsimd::avx2, const float32*,
                                                                                     const float32*, const uint32*,
                                                                                     float32*, uint32, float32,
                                                                                     float32);
        extern template void calculateOutputWiseScoresWeighted<xsimd::avx2, float64>(xsimd::avx2, const float64*,
                                                                                     const float64*, const uint32*,
                                                                                     float64*, uint32, float32,
                                                                                     float32);
        extern template void calculateOutputWiseScoresWeighted<xsimd::avx512f, float32>(xsimd::avx512f, const float32*,
                                                                                        const float32*, const uint32*,
                                                                                        float32*, uint32, float32,
                                                                                        float32);
        extern template void calculateOutputWiseScoresWeighted<xsimd::avx512f, float64>(xsimd::avx512f, const float64*,
                                                                                        const float64*, const uint32*,
                                                                                        float64*, uint32, float32,
                                                                                        float32);
    #endif
    }
}
#endif
