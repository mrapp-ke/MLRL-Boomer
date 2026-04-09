/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/rule_evaluation/simd/functions/calculate_output_wise_qualities.hpp"
#include "mlrl/boosting/rule_evaluation/simd/functions/calculate_output_wise_scores.hpp"

#if SIMD_SUPPORT_ENABLED
namespace boosting {

    /**
     * Implements basic operations for calculating with arrays of gradients and Hessians using single instruction,
     * multiple data (SIMD) operations.
     */
    struct SimdDecomposableVectorMath {
        public:

            /**
             * Calculates the optimal scores to be predicted for several outputs, based on the corresponding gradients
             * and Hessians and taking L1 and L2 regularization into account, and writes them to an output array.
             *
             * @tparam StatisticType            The type of the gradients and Hessians
             * @param gradients                 A pointer to an array that store the gradients that correspond to
             *                                  individual outputs
             * @param hessians                  A pointer to an array that stores the Hessians that corresponds to
             *                                  individual outputs
             * @param outputs                   A pointer to the array into which the optimal scores should be written
             * @param numElements               The number of elements in the arrays `gradients`, `hessians` and
             *                                  `output`
             * @param l1RegularizationWeight    The weight of the L1 regularization
             * @param l2RegularizationWeight    The weight of the L2 regularization
             */
            template<typename StatisticType>
            static inline constexpr void calculateOutputWiseScores(const StatisticType* gradients,
                                                                   const StatisticType* hessians,
                                                                   StatisticType* outputs, uint32 numElements,
                                                                   float32 l1RegularizationWeight,
                                                                   float32 l2RegularizationWeight) {
                auto dispatched = xsimd::dispatch<util::simd_architectures>([&](auto arch) {
                    simd::calculateOutputWiseScores(arch, gradients, hessians, outputs, numElements,
                                                    l1RegularizationWeight, l2RegularizationWeight);
                });
                dispatched();
            }

            /**
             * Calculates the qualities of predictions for several outputs, taking L1 and L2 regularization into
             * account, and the overall quality aggregated over all predictions.
             *
             * @tparam StatisticType            The type of the gradients and Hessians
             * @param scores                    A pointer to an array that stores the predictions for individual outputs
             * @param gradients                 A pointer to an array that stores the gradients that correspond to
             *                                  individual outputs
             * @param hessians                  A pointer to an array that stores the Hessians that correspond to
             *                                  individual outputs
             * @param numElements               The number of elements in the array `scores`, `gradients` and `hessians`
             * @param l1RegularizationWeight    The weight of the L1 regularization
             * @param l2RegularizationWeight    The weight of the L2 regularization
             * @return                          The overall quality
             */
            template<typename StatisticType>
            static inline StatisticType calculateOutputWiseQualities(const StatisticType* scores,
                                                                     const StatisticType* gradients,
                                                                     const StatisticType* hessians, uint32 numElements,
                                                                     float32 l1RegularizationWeight,
                                                                     float32 l2RegularizationWeight) {
                auto dispatched = xsimd::dispatch<util::simd_architectures>([&](auto arch) {
                    return simd::calculateOutputWiseQualities(arch, scores, gradients, hessians, numElements,
                                                              l1RegularizationWeight, l2RegularizationWeight);
                });
                return dispatched();
            }
    };
}
#endif
