/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/rule_evaluation/score_vector_dense.hpp"
#include "rule_evaluation_decomposable_common.hpp"

namespace boosting {

    /**
     * Allows to calculate the predictions of complete rules, as well as their overall quality, based on the gradients
     * and Hessians that are stored by a vector using L1 and L2 regularization.
     *
     * @tparam StatisticVector  The type of the vector that provides access to the gradients and Hessians
     * @tparam IndexVector      The type of the vector that provides access to the indices of the outputs for which
     *                          predictions should be calculated
     */
    template<typename StatisticVector, typename IndexVector>
    class DecomposableCompleteRuleEvaluation final : public IRuleEvaluation<StatisticVector> {
        private:

            typedef typename StatisticVector::statistic_type statistic_type;

            DenseScoreVector<statistic_type, IndexVector> scoreVector_;

            const float32 l1RegularizationWeight_;

            const float32 l2RegularizationWeight_;

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
                uint32 numElements = statisticVector.getNumElements();
                typename StatisticVector::const_iterator statisticIterator = statisticVector.cbegin();
                typename DenseScoreVector<statistic_type, IndexVector>::value_iterator valueIterator =
                  scoreVector_.values_begin();
                statistic_type quality = 0;

                for (uint32 i = 0; i < numElements; i++) {
                    const typename StatisticVector::value_type& statistic = statisticIterator[i];
                    statistic_type predictedScore = calculateOutputWiseScore(
                      statistic.gradient, statistic.hessian, l1RegularizationWeight_, l2RegularizationWeight_);
                    valueIterator[i] = predictedScore;
                    quality += calculateOutputWiseQuality(predictedScore, statistic.gradient, statistic.hessian,
                                                          l1RegularizationWeight_, l2RegularizationWeight_);
                }

                scoreVector_.quality = quality;
                return scoreVector_;
            }
    };

}
