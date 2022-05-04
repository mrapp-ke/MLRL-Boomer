/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/rule_evaluation/score_vector_dense.hpp"
#include "rule_evaluation_label_wise_common.hpp"


namespace boosting {

    /**
     * Allows to calculate the predictions of complete rules, as well as an overall quality score, based on the
     * gradients and Hessians that are stored by a `DenseLabelWiseStatisticVector` using L1 and L2 regularization.
     *
     * @tparam IndexVector The type of the vector that provides access to the labels for which predictions should be
     *                     calculated
     */
    template<typename IndexVector>
    class DenseLabelWiseCompleteRuleEvaluation final : public IRuleEvaluation<DenseLabelWiseStatisticVector> {

        private:

            DenseScoreVector<IndexVector> scoreVector_;

            float64 l1RegularizationWeight_;

            float64 l2RegularizationWeight_;

        public:

            /**
             * @param labelIndices              A reference to an object of template type `IndexVector` that provides
             *                                  access to the indices of the labels for which the rules may predict
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             */
            DenseLabelWiseCompleteRuleEvaluation(const IndexVector& labelIndices, float64 l1RegularizationWeight,
                                                 float64 l2RegularizationWeight)
                : scoreVector_(DenseScoreVector<IndexVector>(labelIndices, true)),
                  l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight) {

            }

            const IScoreVector& calculatePrediction(DenseLabelWiseStatisticVector& statisticVector) override {
                uint32 numElements = statisticVector.getNumElements();
                DenseLabelWiseStatisticVector::const_iterator statisticIterator = statisticVector.cbegin();
                typename DenseScoreVector<IndexVector>::score_iterator scoreIterator = scoreVector_.scores_begin();
                float64 overallQualityScore = 0;

                for (uint32 i = 0; i < numElements; i++) {
                    const Tuple<float64>& tuple = statisticIterator[i];
                    float64 predictedScore = calculateLabelWiseScore(tuple.first, tuple.second, l1RegularizationWeight_,
                                                                    l2RegularizationWeight_);
                    scoreIterator[i] = predictedScore;
                    overallQualityScore += calculateLabelWiseQualityScore(predictedScore, tuple.first, tuple.second,
                                                                          l1RegularizationWeight_,
                                                                          l2RegularizationWeight_);
                }

                scoreVector_.overallQualityScore = overallQualityScore;
                return scoreVector_;
            }

    };

}