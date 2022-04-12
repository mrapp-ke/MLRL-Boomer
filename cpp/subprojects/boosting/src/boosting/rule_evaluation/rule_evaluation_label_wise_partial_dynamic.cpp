#include "boosting/rule_evaluation/rule_evaluation_label_wise_partial_dynamic.hpp"
#include "rule_evaluation_label_wise_complete_common.hpp"


namespace boosting {

    /**
     * Allows to calculate the predictions of partial rules, which predict for a subset of the available labels that is
     * determined dynamically, as well as an overall quality score, based on the gradients and Hessians that are stored
     * by a `DenseLabelWiseStatisticVector` using L1 and L2 regularization.
     *
     * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
     */
    template<typename T>
    class DenseLabelWiseDynamicPartialRuleEvaluation final : public IRuleEvaluation<DenseLabelWiseStatisticVector> {

        private:

            const T& labelIndices_;

            PartialIndexVector indexVector_;

            DenseScoreVector<PartialIndexVector> scoreVector_;

            float64 threshold_;

            float64 l1RegularizationWeight_;

            float64 l2RegularizationWeight_;

        public:

            /**
             * @param labelIndices              A reference to an object of template type `T` that provides access to
             *                                  the indices of the labels for which the rules may predict
             * @param threshold                 A threshold that affects for how many labels the rule heads should
             *                                  predict
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             */
            DenseLabelWiseDynamicPartialRuleEvaluation(const T& labelIndices, float32 threshold,
                                                       float64 l1RegularizationWeight, float64 l2RegularizationWeight)
                : labelIndices_(labelIndices), indexVector_(PartialIndexVector(labelIndices.getNumElements())),
                  scoreVector_(DenseScoreVector<PartialIndexVector>(indexVector_, true)), threshold_(1.0 - threshold),
                  l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight) {

            }

            const IScoreVector& calculatePrediction(DenseLabelWiseStatisticVector& statisticVector) override {
                uint32 numElements = statisticVector.getNumElements();
                DenseLabelWiseStatisticVector::const_iterator statisticIterator = statisticVector.cbegin();
                const Tuple<float64>& firstTuple = statisticIterator[0];
                float64 bestScore = calculateLabelWiseScore(firstTuple.first, firstTuple.second,
                                                            l1RegularizationWeight_, l2RegularizationWeight_);

                for (uint32 i = 1; i < numElements; i++) {
                    const Tuple<float64>& tuple = statisticIterator[i];
                    float64 score = calculateLabelWiseScore(tuple.first, tuple.second, l1RegularizationWeight_,
                                                            l2RegularizationWeight_);

                    if (std::abs(score) > std::abs(bestScore)) {
                        bestScore = score;
                    }
                }

                PartialIndexVector::iterator indexIterator = indexVector_.begin();
                DenseScoreVector<PartialIndexVector>::score_iterator scoreIterator = scoreVector_.scores_begin();
                typename T::const_iterator labelIndexIterator = labelIndices_.cbegin();
                float64 threshold = (bestScore * bestScore) * threshold_;
                float64 overallQualityScore = 0;
                uint32 n = 0;

                for (uint32 i = 0; i < numElements; i++) {
                    const Tuple<float64>& tuple = statisticIterator[i];
                    float64 score = calculateLabelWiseScore(tuple.first, tuple.second, l1RegularizationWeight_,
                                                            l2RegularizationWeight_);

                    if (score * score > threshold) {
                        indexIterator[n] = labelIndexIterator[i];
                        scoreIterator[n] = score;
                        overallQualityScore += calculateLabelWiseQualityScore(score, tuple.first, tuple.second,
                                                                              l1RegularizationWeight_,
                                                                              l2RegularizationWeight_);
                        n++;
                    }
                }

                indexVector_.setNumElements(n, false);
                scoreVector_.overallQualityScore = overallQualityScore;
                return scoreVector_;
            }

    };

    LabelWiseDynamicPartialRuleEvaluationFactory::LabelWiseDynamicPartialRuleEvaluationFactory(
            float32 threshold, float64 l1RegularizationWeight, float64 l2RegularizationWeight)
        : threshold_(threshold), l1RegularizationWeight_(l1RegularizationWeight),
          l2RegularizationWeight_(l2RegularizationWeight) {

    }

    std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> LabelWiseDynamicPartialRuleEvaluationFactory::create(
            const DenseLabelWiseStatisticVector& statisticVector, const CompleteIndexVector& indexVector) const {
        return std::make_unique<DenseLabelWiseDynamicPartialRuleEvaluation<CompleteIndexVector>>(
            indexVector, threshold_, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> LabelWiseDynamicPartialRuleEvaluationFactory::create(
            const DenseLabelWiseStatisticVector& statisticVector, const PartialIndexVector& indexVector) const {
        return std::make_unique<DenseLabelWiseCompleteRuleEvaluation<PartialIndexVector>>(indexVector,
                                                                                          l1RegularizationWeight_,
                                                                                          l2RegularizationWeight_);
    }

}
