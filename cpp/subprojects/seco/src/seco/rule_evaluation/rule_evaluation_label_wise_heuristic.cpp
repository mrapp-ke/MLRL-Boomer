#include "seco/rule_evaluation/rule_evaluation_label_wise_heuristic.hpp"
#include "common/rule_evaluation/score_vector_label_wise_dense.hpp"
#include "../data/confusion_matrices.hpp"


namespace seco {

    /**
     * Allows to calculate the predictions of rules, as well as corresponding quality scores, such that they optimize a
     * heuristic that is applied using label-wise averaging.
     *
     * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
     */
    template<class T>
    class HeuristicLabelWiseRuleEvaluation final : public ILabelWiseRuleEvaluation {

        private:

            std::shared_ptr<IHeuristic> heuristicPtr_;

            bool predictMajority_;

            DenseLabelWiseScoreVector<T> scoreVector_;

        public:

            /**
             * @param labelIndices      A reference to an object of template type `T` that provides access to the
             *                          indices of the labels for which the rules may predict
             * @param heuristicPtr      A shared pointer to an object of type `IHeuristic`, representing the heuristic
             *                          to be optimized
             * @param predictMajority   True, if for each label the majority label should be predicted, false, if the
             *                          minority label should be predicted
             */
            HeuristicLabelWiseRuleEvaluation(const T& labelIndices, std::shared_ptr<IHeuristic> heuristicPtr,
                                             bool predictMajority)
                : heuristicPtr_(heuristicPtr), predictMajority_(predictMajority),
                  scoreVector_(DenseLabelWiseScoreVector<T>(labelIndices)) {

            }

            const ILabelWiseScoreVector& calculateLabelWisePrediction(
                    const DenseVector<uint8>& majorityLabelVector,
                    const DenseConfusionMatrixVector& confusionMatricesTotal,
                    const DenseConfusionMatrixVector& confusionMatricesSubset,
                    const DenseConfusionMatrixVector& confusionMatricesCovered, bool uncovered) override {
                uint32 numPredictions = scoreVector_.getNumElements();
                typename DenseLabelWiseScoreVector<T>::score_iterator scoreIterator = scoreVector_.scores_begin();
                typename DenseLabelWiseScoreVector<T>::index_const_iterator indexIterator =
                    scoreVector_.indices_cbegin();
                typename DenseLabelWiseScoreVector<T>::quality_score_iterator qualityScoreIterator =
                    scoreVector_.quality_scores_begin();
                DenseVector<uint8>::const_iterator majorityIterator = majorityLabelVector.cbegin();
                float64 overallQualityScore = 0;

                for (uint32 i = 0; i < numPredictions; i++) {
                    uint32 index = indexIterator[i];

                    // Set the score to be predicted for the current label...
                    uint8 majorityLabel = majorityIterator[index];
                    float64 score = (float64) (predictMajority_ ? majorityLabel : (majorityLabel ? 0 : 1));
                    scoreIterator[i] = score;

                    // Calculate the quality score for the current label...
                    DenseConfusionMatrixVector::const_iterator coveredIterator =
                        confusionMatricesCovered.confusion_matrix_cbegin(i);
                    DenseConfusionMatrixVector::const_iterator totalIterator =
                        confusionMatricesTotal.confusion_matrix_cbegin(index);

                    uint32 cin = coveredIterator[IN];
                    uint32 cip = coveredIterator[IP];
                    uint32 crn = coveredIterator[RN];
                    uint32 crp = coveredIterator[RP];
                    uint32 uin, uip, urn, urp;

                    if (uncovered) {
                        DenseConfusionMatrixVector::const_iterator subsetIterator =
                            confusionMatricesSubset.confusion_matrix_cbegin(index);
                        uin = cin + totalIterator[IN] - subsetIterator[IN];
                        uip = cip + totalIterator[IP] - subsetIterator[IP];
                        urn = crn + totalIterator[RN] - subsetIterator[RN];
                        urp = crp + totalIterator[RP] - subsetIterator[RP];
                        subsetIterator = confusionMatricesSubset.confusion_matrix_cbegin(i);
                        cin = subsetIterator[IN] - cin;
                        cip = subsetIterator[IP] - cip;
                        crn = subsetIterator[RN] - crn;
                        crp = subsetIterator[RP] - crp;
                    } else {
                        uin = totalIterator[IN] - cin;
                        uip = totalIterator[IP] - cip;
                        urn = totalIterator[RN] - crn;
                        urp = totalIterator[RP] - crp;
                    }

                    score = heuristicPtr_->evaluateConfusionMatrix(cin, cip, crn, crp, uin, uip, urn, urp);
                    qualityScoreIterator[i] = score;
                    overallQualityScore += score;
                }

                overallQualityScore /= numPredictions;
                scoreVector_.overallQualityScore = overallQualityScore;
                return scoreVector_;
            }

    };

    HeuristicLabelWiseRuleEvaluationFactory::HeuristicLabelWiseRuleEvaluationFactory(
            std::shared_ptr<IHeuristic> heuristicPtr, bool predictMajority)
        : heuristicPtr_(heuristicPtr), predictMajority_(predictMajority) {

    }

    std::unique_ptr<ILabelWiseRuleEvaluation> HeuristicLabelWiseRuleEvaluationFactory::create(
            const FullIndexVector& indexVector) const {
        return std::make_unique<HeuristicLabelWiseRuleEvaluation<FullIndexVector>>(indexVector, heuristicPtr_,
                                                                                   predictMajority_);
    }

    std::unique_ptr<ILabelWiseRuleEvaluation> HeuristicLabelWiseRuleEvaluationFactory::create(
            const PartialIndexVector& indexVector) const {
        return std::make_unique<HeuristicLabelWiseRuleEvaluation<PartialIndexVector>>(indexVector, heuristicPtr_,
                                                                                      predictMajority_);
    }

}
