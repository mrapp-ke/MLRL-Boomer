#include "boosting/rule_evaluation/rule_evaluation_label_wise_complete.hpp"
#include "common/rule_evaluation/score_vector_dense.hpp"
#include "common/validation.hpp"
#include "rule_evaluation_label_wise_common.hpp"


namespace boosting {

    /**
     * Calculates and returns an overall quality score that assesses the quality of the scores that are predicted for
     * several labels.
     *
     * @tparam ScoreIterator            The type of the iterator that provides access to the predicted scores
     * @param scoreIterator             An iterator that provides random access to the predicted scores
     * @param statisticIterator         An iterator that provides random access to the gradients and Hessians
     * @param numElements               The number of predicted scores
     * @param l2RegularizationWeight    The weight of the l2 regularization
     * @return                          The overall quality score that has been calculated
     */
    template<typename ScoreIterator>
    static inline constexpr float64 calculateLabelWiseOverallQualityScore(
            ScoreIterator scoreIterator, DenseLabelWiseStatisticVector::const_iterator statisticIterator,
            uint32 numElements, float64 l2RegularizationWeight) {
        float64 overallQualityScore = 0;

        for (uint32 i = 0; i < numElements; i++) {
            const Tuple<float64>& tuple = statisticIterator[i];
            overallQualityScore += calculateLabelWiseQualityScore(scoreIterator[i], tuple.first, tuple.second,
                                                                  l2RegularizationWeight);
        }

        return overallQualityScore;
    }

    /**
     * Allows to calculate the predictions of complete rules, as well as an overall quality score, based on the
     * gradients and Hessians that are stored by a `DenseLabelWiseStatisticVector` using L2 regularization.
     *
     * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
     */
    template<typename T>
    class LabelWiseCompleteRuleEvaluation final : public ILabelWiseRuleEvaluation<DenseLabelWiseStatisticVector> {

        private:

            DenseScoreVector<T> scoreVector_;

            float64 l2RegularizationWeight_;

        public:

            /**
             * @param labelIndices              A reference to an object of template type `T` that provides access to
             *                                  the indices of the labels for which the rules may predict
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             */
            LabelWiseCompleteRuleEvaluation(const T& labelIndices, float64 l2RegularizationWeight)
                : scoreVector_(DenseScoreVector<T>(labelIndices)), l2RegularizationWeight_(l2RegularizationWeight) {

            }

            const ILabelWiseScoreVector& calculateLabelWisePrediction(
                    const DenseLabelWiseStatisticVector& statisticVector) override {

            }

            const IScoreVector& calculatePrediction(const DenseLabelWiseStatisticVector& statisticVector) override {
                uint32 numElements = statisticVector.getNumElements();
                DenseLabelWiseStatisticVector::const_iterator statisticIterator = statisticVector.cbegin();
                typename DenseScoreVector<T>::score_iterator scoreIterator = scoreVector_.scores_begin();

                for (uint32 i = 0; i < numElements; i++) {
                    const Tuple<float64>& tuple = statisticIterator[i];
                    scoreIterator[i] = calculateLabelWiseScore(tuple.first, tuple.second, l2RegularizationWeight_);
                }

                scoreVector_.overallQualityScore = calculateLabelWiseOverallQualityScore(scoreIterator,
                                                                                         statisticIterator, numElements,
                                                                                         l2RegularizationWeight_);
                return scoreVector_;
            }

    };

    LabelWiseCompleteRuleEvaluationFactory::LabelWiseCompleteRuleEvaluationFactory(float64 l2RegularizationWeight)
        : l2RegularizationWeight_(l2RegularizationWeight) {
        assertGreaterOrEqual<float64>("l2RegularizationWeight", l2RegularizationWeight, 0);
    }

    std::unique_ptr<ILabelWiseRuleEvaluation<DenseLabelWiseStatisticVector>> LabelWiseCompleteRuleEvaluationFactory::createDense(
            const CompleteIndexVector& indexVector) const {
        return std::make_unique<LabelWiseCompleteRuleEvaluation<CompleteIndexVector>>(indexVector,
                                                                                      l2RegularizationWeight_);
    }

    std::unique_ptr<ILabelWiseRuleEvaluation<DenseLabelWiseStatisticVector>> LabelWiseCompleteRuleEvaluationFactory::createDense(
            const PartialIndexVector& indexVector) const {
        return std::make_unique<LabelWiseCompleteRuleEvaluation<PartialIndexVector>>(indexVector,
                                                                                     l2RegularizationWeight_);
    }

}