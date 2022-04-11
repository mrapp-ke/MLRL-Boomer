#include "boosting/rule_evaluation/rule_evaluation_label_wise_partial_fixed.hpp"
#include "rule_evaluation_label_wise_partial_fixed_common.hpp"
#include "rule_evaluation_label_wise_complete_common.hpp"


namespace boosting {

    /**
     * Allows to calculate the predictions of partial rules that predict for a predefined number of labels, as well as
     * an overall quality score, based on the gradients and Hessians that are stored by a
     * `DenseLabelWiseStatisticVector` using L1 and L2 regularization.
     *
     * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
     */
    template<typename T>
    class DenseLabelWiseFixedPartialRuleEvaluation final : public IRuleEvaluation<DenseLabelWiseStatisticVector> {

        private:

            const T& labelIndices_;

            PartialIndexVector indexVector_;

            DenseScoreVector<PartialIndexVector> scoreVector_;

            float64 l1RegularizationWeight_;

            float64 l2RegularizationWeight_;

            PriorityQueue priorityQueue_;

        public:

            /**
             * @param labelIndices              A reference to an object of template type `T` that provides access to
             *                                  the indices of the labels for which the rules may predict
             * @param numPredictions            The number of labels for which the rules should predict
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             */
            DenseLabelWiseFixedPartialRuleEvaluation(const T& labelIndices, uint32 numPredictions,
                                                     float64 l1RegularizationWeight, float64 l2RegularizationWeight)
                : labelIndices_(labelIndices), indexVector_(PartialIndexVector(numPredictions)),
                  scoreVector_(DenseScoreVector<PartialIndexVector>(indexVector_, false)),
                  l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight) {

            }

            const IScoreVector& calculatePrediction(DenseLabelWiseStatisticVector& statisticVector) override {
                uint32 numElements = statisticVector.getNumElements();
                uint32 numPredictions = indexVector_.getNumElements();
                DenseLabelWiseStatisticVector::const_iterator statisticIterator = statisticVector.cbegin();
                typename T::const_iterator labelIndexIterator = labelIndices_.cbegin();
                sortLabelWiseQualityScores(priorityQueue_, numPredictions, statisticIterator, labelIndexIterator,
                                           numElements, l1RegularizationWeight_, l2RegularizationWeight_);
                PartialIndexVector::iterator indexIterator = indexVector_.begin();
                DenseScoreVector<PartialIndexVector>::score_iterator scoreIterator = scoreVector_.scores_begin();
                float64 overallQualityScore = 0;

                for (uint32 i = 0; i < numPredictions; i++) {
                    const IndexedValue<float64>& entry = priorityQueue_.top();
                    uint32 index = entry.index;
                    float64 predictedScore = entry.value;
                    indexIterator[i] = index;
                    scoreIterator[i] = predictedScore;
                    const Tuple<float64>& tuple = statisticIterator[index];
                    overallQualityScore += calculateLabelWiseQualityScore(predictedScore, tuple.first, tuple.second,
                                                                          l1RegularizationWeight_,
                                                                          l2RegularizationWeight_);
                    priorityQueue_.pop();
                }

                scoreVector_.overallQualityScore = overallQualityScore;
                return scoreVector_;
            }

    };

    LabelWiseFixedPartialRuleEvaluationFactory::LabelWiseFixedPartialRuleEvaluationFactory(
            float32 labelRatio, uint32 minLabels, uint32 maxLabels, float64 l1RegularizationWeight,
            float64 l2RegularizationWeight)
        : labelRatio_(labelRatio), minLabels_(minLabels), maxLabels_(maxLabels),
          l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight) {

    }

    std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> LabelWiseFixedPartialRuleEvaluationFactory::create(
            const DenseLabelWiseStatisticVector& statisticVector, const CompleteIndexVector& indexVector) const {
        uint32 numPredictions = calculateBoundedFraction(indexVector.getNumElements(), labelRatio_, minLabels_,
                                                         maxLabels_);
        return std::make_unique<DenseLabelWiseFixedPartialRuleEvaluation<CompleteIndexVector>>(indexVector,
                                                                                               numPredictions,
                                                                                               l1RegularizationWeight_,
                                                                                               l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> LabelWiseFixedPartialRuleEvaluationFactory::create(
            const DenseLabelWiseStatisticVector& statisticVector, const PartialIndexVector& indexVector) const {
        return std::make_unique<DenseLabelWiseCompleteRuleEvaluation<PartialIndexVector>>(indexVector,
                                                                                          l1RegularizationWeight_,
                                                                                          l2RegularizationWeight_);
    }

}
