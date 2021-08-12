#include "seco/rule_evaluation/rule_evaluation_label_wise_majority.hpp"
#include "common/rule_evaluation/score_vector_dense.hpp"
#include "../data/confusion_matrices.hpp"


namespace seco {

    template<typename IndexIterator, typename LabelIterator, typename ScoreIterator>
    static inline void calculateLabelWiseScores(IndexIterator indexIterator, LabelIterator labelIterator,
                                                ScoreIterator scoreIterator, uint32 numElements) {
        uint32 previousIndex = 0;

        for (uint32 i = 0; i < numElements; i++) {
            uint32 index = indexIterator[i];
            std::advance(labelIterator, index - previousIndex);
            scoreIterator[i] = (float64) *labelIterator;
            previousIndex = index;
        }
    }

    template<typename IndexIterator>
    static inline float64 calculateOverallQualityScore(IndexIterator indexIterator,
                                                       const DenseConfusionMatrixVector& confusionMatricesTotal,
                                                       const DenseConfusionMatrixVector& confusionMatricesSubset,
                                                       const DenseConfusionMatrixVector& confusionMatricesCovered,
                                                       bool uncovered, const IHeuristic& heuristic,
                                                       uint32 numElements) {
        float64 sumOfQualityScores = 0;

        for (uint32 i = 0; i < numElements; i++) {
            uint32 index = indexIterator[i];
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

            sumOfQualityScores += heuristic.evaluateConfusionMatrix(cin, cip, crn, crp, uin, uip, urn, urp);
        }

        return sumOfQualityScores / numElements;
    }

    /**
     * Allows to calculate the predictions of rules, as well as corresponding quality scores, such that they predict
     * each label as relevant or irrelevant, depending on whether it is associated with the majority of the training
     * examples or not.
     *
     * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
     */
    template<typename T>
    class LabelWiseMajorityRuleEvaluation final : public IRuleEvaluation {

        private:

            DenseScoreVector<T> scoreVector_;

            const IHeuristic& heuristic_;

        public:

            /**
             * @param labelIndices A reference to an object of template type `T` that provides access to the indices of
             *                     the labels for which the rules may predict
             */
            LabelWiseMajorityRuleEvaluation(const T& labelIndices, const IHeuristic& heuristic)
                : scoreVector_(DenseScoreVector<T>(labelIndices)), heuristic_(heuristic) {

            }

            const ILabelWiseScoreVector& calculateLabelWisePrediction(
                    const BinarySparseArrayVector& majorityLabelVector,
                    const DenseConfusionMatrixVector& confusionMatricesTotal,
                    const DenseConfusionMatrixVector& confusionMatricesSubset,
                    const DenseConfusionMatrixVector& confusionMatricesCovered, bool uncovered) override {

            }

            const IScoreVector& calculatePrediction(const BinarySparseArrayVector& majorityLabelVector,
                                                    const DenseConfusionMatrixVector& confusionMatricesTotal,
                                                    const DenseConfusionMatrixVector& confusionMatricesSubset,
                                                    const DenseConfusionMatrixVector& confusionMatricesCovered,
                                                    bool uncovered) override {
                typename DenseScoreVector<T>::score_iterator scoreIterator = scoreVector_.scores_begin();
                typename DenseScoreVector<T>::index_const_iterator indexIterator = scoreVector_.indices_cbegin();
                auto labelIterator = make_index_forward_iterator(majorityLabelVector.indices_cbegin(),
                                                                 majorityLabelVector.indices_cend());
                uint32 numElements = scoreVector_.getNumElements();

                // Calculate the prediction for each label...
                calculateLabelWiseScores(indexIterator, labelIterator, scoreIterator, numElements);

                // Calculate an overall quality score...
                scoreVector_.overallQualityScore = calculateOverallQualityScore(indexIterator, confusionMatricesTotal,
                                                                                confusionMatricesSubset,
                                                                                confusionMatricesCovered, uncovered,
                                                                                heuristic_, numElements);

                return scoreVector_;
            }

    };

    LabelWiseMajorityRuleEvaluationFactory::LabelWiseMajorityRuleEvaluationFactory(
            std::unique_ptr<IHeuristic> heuristicPtr)
        : heuristicPtr_(std::move(heuristicPtr)) {

    }

    std::unique_ptr<IRuleEvaluation> LabelWiseMajorityRuleEvaluationFactory::create(
            const CompleteIndexVector& indexVector) const {
        return std::make_unique<LabelWiseMajorityRuleEvaluation<CompleteIndexVector>>(indexVector, *heuristicPtr_);
    }

    std::unique_ptr<IRuleEvaluation> LabelWiseMajorityRuleEvaluationFactory::create(
            const PartialIndexVector& indexVector) const {
        return std::make_unique<LabelWiseMajorityRuleEvaluation<PartialIndexVector>>(indexVector, *heuristicPtr_);
    }

}
