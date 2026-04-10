/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/rule_evaluation/scalar_math_decomposable.hpp"
#include "mlrl/boosting/rule_evaluation/simd/functions/calculate_output_wise_qualities.hpp"

#if SIMD_SUPPORT_ENABLED
namespace boosting {
    namespace simd {

        template<typename Arch, typename StatisticType>
        StatisticType calculateOutputWiseQualities(Arch, const StatisticType* scores, const StatisticType* gradients,
                                                   const StatisticType* hessians, uint32 numElements,
                                                   float32 l1RegularizationWeight, float32 l2RegularizationWeight) {
            using batch = xsimd::batch<StatisticType, Arch>;
            const batch l1Weight = batch(l1RegularizationWeight);
            const batch l2Weight = batch(l2RegularizationWeight);
            const batch half = batch(0.5);
            constexpr std::size_t batchSize = batch::size;
            uint32 batchEnd = numElements - (numElements % batchSize);
            uint32 i = 0;
            StatisticType overallQuality = 0;

            for (; i < batchEnd; i += batchSize) {
                batch batchScores = batch::load_unaligned(scores + i);
                batch batchGradients = batch::load_unaligned(gradients + i);
                batch batchHessians = batch::load_unaligned(hessians + i);
                batch scorePow = batchScores * batchScores;
                batch l1Term = l1Weight * xsimd::abs(batchScores);
                batch l2Term = half * l2Weight * scorePow;
                batch qualities = (batchGradients * batchScores) + (half * batchHessians * scorePow) + l1Term + l2Term;
                overallQuality += xsimd::reduce_add(qualities);
            }

            for (; i < numElements; i++) {
                overallQuality += calculateOutputWiseQuality(scores[i], gradients[i], hessians[i],
                                                             l1RegularizationWeight, l2RegularizationWeight);
            }

            return overallQuality;
        }
    }
}
#endif
