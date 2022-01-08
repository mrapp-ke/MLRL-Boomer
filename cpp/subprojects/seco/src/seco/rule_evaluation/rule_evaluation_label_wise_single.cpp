#include "seco/rule_evaluation/rule_evaluation_label_wise_single.hpp"
#include "common/indices/index_vector_partial.hpp"
#include "common/rule_evaluation/score_vector_dense.hpp"
#include "common/util/validation.hpp"
#include "rule_evaluation_label_wise_common.hpp"


namespace seco {

    /**
     * Allows to calculate the predictions of single-label rules, as well as corresponding quality scores, such that
     * they optimize a heuristic that is applied using label-wise averaging.
     *
     * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
     */
    template<typename T>
    class LabelWiseSingleLabelRuleEvaluation final : public IRuleEvaluation {

        private:

            const T& labelIndices_;

            PartialIndexVector indexVector_;

            DenseScoreVector<PartialIndexVector> scoreVector_;

            std::unique_ptr<IHeuristic> heuristicPtr_;

        public:

            /**
             * @param labelIndices  A reference to an object of template type `T` that provides access to the indices of
             *                      the labels for which the rules may predict
             * @param heuristicPtr  An unique pointer to an object of type `IHeuristic` that implements the heuristic to
             *                      be optimized
             */
            LabelWiseSingleLabelRuleEvaluation(const T& labelIndices, std::unique_ptr<IHeuristic> heuristicPtr)
                : labelIndices_(labelIndices), indexVector_(PartialIndexVector(1)),
                  scoreVector_(DenseScoreVector<PartialIndexVector>(indexVector_)),
                  heuristicPtr_(std::move(heuristicPtr)) {

            }

            const IScoreVector& calculatePrediction(
                    const VectorConstView<uint32>& majorityLabelIndices,
                    const DenseConfusionMatrixVector& confusionMatricesTotal,
                    const DenseConfusionMatrixVector& confusionMatricesCovered) override {
                uint32 numElements = labelIndices_.getNumElements();
                typename T::const_iterator indexIterator = labelIndices_.cbegin();
                DenseConfusionMatrixVector::const_iterator totalIterator = confusionMatricesTotal.cbegin();
                DenseConfusionMatrixVector::const_iterator coveredIterator = confusionMatricesCovered.cbegin();
                uint32 bestIndex = indexIterator[0];
                float64 bestQualityScore = calculateLabelWiseQualityScore(totalIterator[bestIndex], coveredIterator[0],
                                                                          *heuristicPtr_);

                for (uint32 i = 1; i < numElements; i++) {
                    uint32 index = indexIterator[i];
                    float64 qualityScore = calculateLabelWiseQualityScore(totalIterator[index], coveredIterator[i],
                                                                          *heuristicPtr_);

                    if (qualityScore < bestQualityScore) {
                        bestIndex = index;
                        bestQualityScore = qualityScore;
                    }
                }

                auto labelIterator = make_binary_forward_iterator(majorityLabelIndices.cbegin(),
                                                                  majorityLabelIndices.cend());
                std::advance(labelIterator, bestIndex);
                scoreVector_.scores_begin()[0] = (float64) !(*labelIterator);
                indexVector_.begin()[0] = bestIndex;
                scoreVector_.overallQualityScore = bestQualityScore;
                return scoreVector_;
            }

    };

    LabelWiseSingleLabelRuleEvaluationFactory::LabelWiseSingleLabelRuleEvaluationFactory(
            std::unique_ptr<IHeuristicFactory> heuristicFactoryPtr)
        : heuristicFactoryPtr_(std::move(heuristicFactoryPtr)) {
        assertNotNull("heuristicFactoryPtr", heuristicFactoryPtr_.get());
    }

    std::unique_ptr<IRuleEvaluation> LabelWiseSingleLabelRuleEvaluationFactory::create(
            const CompleteIndexVector& indexVector) const {
        std::unique_ptr<IHeuristic> heuristicPtr = heuristicFactoryPtr_->create();
        return std::make_unique<LabelWiseSingleLabelRuleEvaluation<CompleteIndexVector>>(indexVector,
                                                                                         std::move(heuristicPtr));
    }

    std::unique_ptr<IRuleEvaluation> LabelWiseSingleLabelRuleEvaluationFactory::create(
            const PartialIndexVector& indexVector) const {
        std::unique_ptr<IHeuristic> heuristicPtr = heuristicFactoryPtr_->create();
        return std::make_unique<LabelWiseSingleLabelRuleEvaluation<PartialIndexVector>>(indexVector,
                                                                                        std::move(heuristicPtr));
    }

}
