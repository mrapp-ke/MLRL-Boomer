/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/math/vector_math.hpp"
#include "scalar_math_decomposable.hpp"

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
                    StatisticType gradient = gradients[i];
                    outputs[i] =
                      math::divideOrZero(-gradient + getL1RegularizationWeight(gradient, l1RegularizationWeight),
                                         hessians[i] + l2RegularizationWeight);
                }
            }

            /**
             * Calculates and returns the optimal score to be predicted for a single output, based on the
             * corresponding gradient and Hessian and taking L1 and L2 regularization into account.
             *
             * @tparam StatisticType            The type of the gradient and Hessian
             * @param gradient                  The gradient that corresponds to the output
             * @param hessian                   The Hessian that corresponds to the output
             * @param l1RegularizationWeight    The weight of the L1 regularization
             * @param l2RegularizationWeight    The weight of the L2 regularization
             * @return                          The predicted score that has been calculated
             */
            template<typename StatisticType>
            static inline constexpr StatisticType calculateOutputWiseScore(StatisticType gradient,
                                                                           StatisticType hessian,
                                                                           float32 l1RegularizationWeight,
                                                                           float32 l2RegularizationWeight) {
                return math::divideOrZero(-gradient + getL1RegularizationWeight(gradient, l1RegularizationWeight),
                                          hessian + l2RegularizationWeight);
            }

            /**
             * Calculates the qualities of predictions for several outputs, taking L1 and L2 regularization into
             * account, and writes them to an output array. In addition, the overall quality aggegregated over all
             * predictions is returned.
             *
             * @tparam StatisticType            The type of the gradients and Hessians
             * @param scores                    A pointer to an array that stores the predictions for individual outputs
             * @param gradients                 A pointer to an array that stores the gradients that correspond to
             *                                  individual outputs
             * @param hessians                  A pointer to an array that stores the Hessians that correspond to
             *                                  individual outputs
             * @param outputs                   A pointer to the array into which the qualities should be written
             * @param l1RegularizationWeight    The weight of the L1 regularization
             * @param l2RegularizationWeight    The weight of the L2 regularization
             * @return                          The overal quality
             */
            template<typename StatisticType>
            static inline StatisticType calculateOutputWiseQualities(const StatisticType* scores,
                                                                     const StatisticType* gradients,
                                                                     const StatisticType* hessians,
                                                                     StatisticType* outputs, uint32 numElements,
                                                                     float32 l1RegularizationWeight,
                                                                     float32 l2RegularizationWeight) {
                StatisticType overallQuality = 0;

                for (uint32 i = 0; i < numElements; i++) {
                    StatisticType score = scores[i];
                    StatisticType scorePow = score * score;
                    StatisticType quality = (gradients[i] * score) + (0.5 * hessians[i] * scorePow);
                    StatisticType l1RegularizationTerm = l1RegularizationWeight * std::abs(score);
                    StatisticType l2RegularizationTerm = 0.5 * l2RegularizationWeight * scorePow;
                    quality += l1RegularizationTerm + l2RegularizationTerm;
                    outputs[i] = quality;
                    overallQuality += quality;
                }

                return overallQuality;
            }

            /**
             * Calculates and returns the quality of the prediction for a single output, taking L1 and L2
             * regularization into account.
             *
             * @tparam StatisticType            The type of the predicted score, gradient and Hessian
             * @param score                     The predicted score
             * @param gradient                  The gradient
             * @param hessian                   The Hessian
             * @param l1RegularizationWeight    The weight of the L1 regularization
             * @param l2RegularizationWeight    The weight of the L2 regularization
             * @return                          The quality that has been calculated
             */
            template<typename StatisticType>
            static inline StatisticType calculateOutputWiseQuality(StatisticType score, StatisticType gradient,
                                                                   StatisticType hessian,
                                                                   float32 l1RegularizationWeight,
                                                                   float32 l2RegularizationWeight) {
                StatisticType scorePow = score * score;
                StatisticType quality = (gradient * score) + (0.5 * hessian * scorePow);
                StatisticType l1RegularizationTerm = l1RegularizationWeight * std::abs(score);
                StatisticType l2RegularizationTerm = 0.5 * l2RegularizationWeight * scorePow;
                return quality + l1RegularizationTerm + l2RegularizationTerm;
            }
    };
}
