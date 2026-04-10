/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/rule_evaluation/scalar_math_decomposable.hpp"
#include "mlrl/boosting/rule_evaluation/simd/functions/calculate_output_wise_scores.hpp"

#if SIMD_SUPPORT_ENABLED
namespace boosting {
    namespace simd {

        template<typename Arch, typename StatisticType>
        void calculateOutputWiseScores(Arch, const StatisticType* gradients, const StatisticType* hessians,
                                       StatisticType* outputs, uint32 numElements, float32 l1RegularizationWeight,
                                       float32 l2RegularizationWeight) {
            using batch = xsimd::batch<StatisticType, Arch>;
            const batch zero = batch(0);
            const batch epsilon = batch(1e-12);
            const batch l1Weight = batch(l1RegularizationWeight);
            const batch l2Weight = batch(l2RegularizationWeight);
            constexpr std::size_t batchSize = batch::size;
            uint32 batchEnd = numElements - (numElements % batchSize);
            uint32 i = 0;

            for (; i < batchEnd; i += batchSize) {
                batch batchGradients = batch::load_unaligned(gradients + i);
                batch batchHessians = batch::load_unaligned(hessians + i);
                batch l1Term = xsimd::select(batchGradients > l1Weight, -l1Weight,
                                             xsimd::select(batchGradients < -l1Weight, l1Weight, zero));
                batch numerator = -batchGradients + l1Term;
                batch denominator = batchHessians + l2Weight;
                batch result = xsimd::select(denominator > epsilon, numerator / denominator, zero);
                result.store_unaligned(outputs + i);
            }

            for (; i < numElements; i++) {
                outputs[i] =
                  calculateOutputWiseScore(gradients[i], hessians[i], l1RegularizationWeight, l2RegularizationWeight);
            }
        }
    }
}
#endif
