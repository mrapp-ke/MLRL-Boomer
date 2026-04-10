/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/math/vector_math.hpp"
#include "mlrl/boosting/rule_evaluation/scalar_math_decomposable.hpp"

namespace boosting {

    /**
     * Implements basic operations for calculating with arrays of gradients and Hessians by applying the respective
     * operations to each element in the arrays sequentially.
     */
    struct SequentialDecomposableVectorMath {
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
                for (uint32 i = 0; i < numElements; i++) {
                    outputs[i] = calculateOutputWiseScore(gradients[i], hessians[i], l1RegularizationWeight,
                                                          l2RegularizationWeight);
                }
            }

            /**
             * Calculates the qualities of predictions for several outputs, taking L1 and L2 regularization into
             * account, and returns the overall quality aggregated over all predictions.
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
            static inline StatisticType aggregateOutputWiseQualities(const StatisticType* scores,
                                                                     const StatisticType* gradients,
                                                                     const StatisticType* hessians, uint32 numElements,
                                                                     float32 l1RegularizationWeight,
                                                                     float32 l2RegularizationWeight) {
                StatisticType overallQuality = 0;

                for (uint32 i = 0; i < numElements; i++) {
                    overallQuality += calculateOutputWiseQuality(scores[i], gradients[i], hessians[i],
                                                                 l1RegularizationWeight, l2RegularizationWeight);
                }

                return overallQuality;
            }
    };
}
