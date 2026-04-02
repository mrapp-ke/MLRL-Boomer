/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/data/vector_statistic_decomposable_dense.hpp"
#include "mlrl/boosting/data/vector_statistic_decomposable_sparse.hpp"
#include "mlrl/common/rule_evaluation/score_vector_dense.hpp"
#include "vector_math_decomposable.hpp"

namespace boosting {

    /**
     * Allows to calculate the predictions of complete rules, as well as their overall quality, based on the gradients
     * and Hessians that are stored by a vector using L1 and L2 regularization.
     *
     * @tparam StatisticVector  The type of the vector that provides access to the gradients and Hessians
     * @tparam IndexVector      The type of the vector that provides access to the indices of the outputs for which
     *                          predictions should be calculated
     * @tparam VectorMath       The type that implements basic operations for calculating with gradients and Hessians
     */
    template<typename StatisticVector, typename IndexVector, typename VectorMath>
    class DecomposableCompleteRuleEvaluation final : public IRuleEvaluation<StatisticVector> {
        private:

            using statistic_type = StatisticVector::statistic_type;

            DenseScoreVector<statistic_type, IndexVector> scoreVector_;

            const float32 l1RegularizationWeight_;

            const float32 l2RegularizationWeight_;

            template<typename StatisticType, typename WeightType>
            static inline void calculateScoresInternally(
              const SparseDecomposableStatisticVectorView<StatisticType, WeightType>& statisticVector,
              DenseScoreVector<StatisticType, IndexVector>& scoreVector, float32 l1RegularizationWeight,
              float32 l2RegularizationWeight) {
                uint32 numElements = statisticVector.getNumElements();
                auto gradientIterator = statisticVector.gradients_cbegin();
                auto hessianIterator = statisticVector.hessians_cbegin();
                auto valueIterator = scoreVector.values_begin();
                statistic_type quality = 0;

                for (uint32 i = 0; i < numElements; i++) {
                    statistic_type gradient = gradientIterator[i];
                    statistic_type hessian = hessianIterator[i];
                    statistic_type predictedScore = VectorMath::calculateOutputWiseScore(
                      gradient, hessian, l1RegularizationWeight, l2RegularizationWeight);
                    valueIterator[i] = predictedScore;
                    quality += VectorMath::calculateOutputWiseQuality(predictedScore, gradient, hessian,
                                                                      l1RegularizationWeight, l2RegularizationWeight);
                }

                scoreVector.quality = quality;
            }

            template<typename StatisticType>
            static inline void calculateScoresInternally(
              const DenseDecomposableStatisticVectorView<StatisticType>& statisticVector,
              DenseScoreVector<StatisticType, IndexVector>& scoreVector, float32 l1RegularizationWeight,
              float32 l2RegularizationWeight) {
                uint32 numElements = statisticVector.getNumElements();
                auto gradientIterator = statisticVector.gradients_cbegin();
                auto hessianIterator = statisticVector.hessians_cbegin();
                auto valueIterator = scoreVector.values_begin();
                VectorMath::calculateOutputWiseScores(gradientIterator, hessianIterator, valueIterator, numElements,
                                                      l1RegularizationWeight, l2RegularizationWeight);
                scoreVector.quality = VectorMath::calculateOutputWiseQualities(
                  valueIterator, gradientIterator, hessianIterator, valueIterator, numElements, l1RegularizationWeight,
                  l2RegularizationWeight);
            }

        public:

            /**
             * @param outputIndices             A reference to an object of template type `IndexVector` that provides
             *                                  access to the indices of the outputs for which the rules may predict
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             */
            DecomposableCompleteRuleEvaluation(const IndexVector& outputIndices, float32 l1RegularizationWeight,
                                               float32 l2RegularizationWeight)
                : scoreVector_(outputIndices, true), l1RegularizationWeight_(l1RegularizationWeight),
                  l2RegularizationWeight_(l2RegularizationWeight) {}

            const IScoreVector& calculateScores(StatisticVector& statisticVector) override {
                calculateScoresInternally(statisticVector, scoreVector_, l1RegularizationWeight_,
                                          l2RegularizationWeight_);
                return scoreVector_;
            }
    };

}
