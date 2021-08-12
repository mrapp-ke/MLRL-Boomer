#include "seco/rule_evaluation/rule_evaluation_label_wise_heuristic.hpp"
#include "common/rule_evaluation/score_vector_label_wise_dense.hpp"
#include "common/validation.hpp"


namespace seco {

    /**
     * Allows to calculate the predictions of rules, as well as corresponding quality scores, such that they optimize a
     * heuristic that is applied using label-wise averaging.
     *
     * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
     */
    template<typename T>
    class HeuristicLabelWiseRuleEvaluation final : public IRuleEvaluation {

        private:

            const IHeuristic& heuristic_;

            bool predictMajority_;

            DenseLabelWiseScoreVector<T> scoreVector_;

        public:

            /**
             * @param labelIndices      A reference to an object of template type `T` that provides access to the
             *                          indices of the labels for which the rules may predict
             * @param heuristic         A reference to an object of type `IHeuristic`, representing the heuristic to be
             *                          optimized
             * @param predictMajority   True, if for each label the majority label should be predicted, false, if the
             *                          minority label should be predicted
             */
            HeuristicLabelWiseRuleEvaluation(const T& labelIndices, const IHeuristic& heuristic, bool predictMajority)
                : heuristic_(heuristic), predictMajority_(predictMajority),
                  scoreVector_(DenseLabelWiseScoreVector<T>(labelIndices)) {

            }

            const ILabelWiseScoreVector& calculateLabelWisePrediction(
                    const BinarySparseArrayVector& majorityLabelVector,
                    const DenseConfusionMatrixVector& confusionMatricesTotal,
                    const DenseConfusionMatrixVector& confusionMatricesSubset,
                    const DenseConfusionMatrixVector& confusionMatricesCovered, bool uncovered) override {
                uint32 numPredictions = scoreVector_.getNumElements();
                typename DenseLabelWiseScoreVector<T>::score_iterator scoreIterator = scoreVector_.scores_begin();
                typename DenseLabelWiseScoreVector<T>::index_const_iterator indexIterator =
                    scoreVector_.indices_cbegin();
                typename DenseLabelWiseScoreVector<T>::quality_score_iterator qualityScoreIterator =
                    scoreVector_.quality_scores_begin();
                auto majorityIterator = make_index_forward_iterator(majorityLabelVector.indices_cbegin(),
                                                                    majorityLabelVector.indices_cend());
                float64 overallQualityScore = 0;
                uint32 previousIndex = 0;

                for (uint32 i = 0; i < numPredictions; i++) {
                    uint32 index = indexIterator[i];

                    // Set the score to be predicted for the current label...
                    std::advance(majorityIterator, index - previousIndex);
                    bool majorityLabel = *majorityIterator;
                    float64 score = (float64) (predictMajority_ ? majorityLabel : !majorityLabel);
                    scoreIterator[i] = score;

                    // Calculate the quality score for the current label...
                    DenseConfusionMatrixVector::const_iterator coveredIterator = confusionMatricesCovered.cbegin();
                    DenseConfusionMatrixVector::const_iterator totalIterator = confusionMatricesTotal.cbegin();
                    const ConfusionMatrix& totalConfusionMatrix = totalIterator[index];
                    ConfusionMatrix coveredConfusionMatrix = coveredIterator[i];
                    ConfusionMatrix uncoveredConfusionMatrix;

                    if (uncovered) {
                        DenseConfusionMatrixVector::const_iterator subsetIterator = confusionMatricesSubset.cbegin();
                        const ConfusionMatrix& subsetConfusionMatrix = subsetIterator[index];
                        coveredConfusionMatrix = subsetConfusionMatrix - coveredConfusionMatrix;
                    }

                    uncoveredConfusionMatrix = totalConfusionMatrix - coveredConfusionMatrix;

                    score = heuristic_.evaluateConfusionMatrix(
                        coveredConfusionMatrix.in, coveredConfusionMatrix.ip, coveredConfusionMatrix.rn,
                        coveredConfusionMatrix.rp, uncoveredConfusionMatrix.in, uncoveredConfusionMatrix.ip,
                        uncoveredConfusionMatrix.rn, uncoveredConfusionMatrix.rp);
                    qualityScoreIterator[i] = score;
                    overallQualityScore += score;
                    previousIndex = index;
                }

                overallQualityScore /= numPredictions;
                scoreVector_.overallQualityScore = overallQualityScore;
                return scoreVector_;
            }

            const IScoreVector& calculatePrediction(const BinarySparseArrayVector& majorityLabelVector,
                                                    const DenseConfusionMatrixVector& confusionMatricesTotal,
                                                    const DenseConfusionMatrixVector& confusionMatricesSubset,
                                                    const DenseConfusionMatrixVector& confusionMatricesCovered,
                                                    bool uncovered) override {
                // TODO Implement
                return scoreVector_;
            }

    };

    HeuristicLabelWiseRuleEvaluationFactory::HeuristicLabelWiseRuleEvaluationFactory(
            std::unique_ptr<IHeuristic> heuristicPtr, bool predictMajority)
        : heuristicPtr_(std::move(heuristicPtr)), predictMajority_(predictMajority) {
        assertNotNull("heuristicPtr", heuristicPtr_.get());
    }

    std::unique_ptr<IRuleEvaluation> HeuristicLabelWiseRuleEvaluationFactory::create(
            const CompleteIndexVector& indexVector) const {
        return std::make_unique<HeuristicLabelWiseRuleEvaluation<CompleteIndexVector>>(indexVector, *heuristicPtr_,
                                                                                       predictMajority_);
    }

    std::unique_ptr<IRuleEvaluation> HeuristicLabelWiseRuleEvaluationFactory::create(
            const PartialIndexVector& indexVector) const {
        return std::make_unique<HeuristicLabelWiseRuleEvaluation<PartialIndexVector>>(indexVector, *heuristicPtr_,
                                                                                      predictMajority_);
    }

}
