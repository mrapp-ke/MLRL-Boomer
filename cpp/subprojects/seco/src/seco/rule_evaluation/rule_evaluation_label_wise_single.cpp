#include "seco/rule_evaluation/rule_evaluation_label_wise_single.hpp"
#include "common/indices/index_vector_partial.hpp"
#include "common/rule_evaluation/score_vector_dense.hpp"


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

            PartialIndexVector indexVector_;

            DenseScoreVector<PartialIndexVector> scoreVector_;

            const IHeuristic& heuristic_;

        public:

            /**
             * @param labelIndices  A reference to an object of template type `T` that provides access to the indices of
             *                      the labels for which the rules may predict
             * @param heuristic     A reference to an object of type `IHeuristic`, implementing the heuristic to be
             *                      optimized
             */
            LabelWiseSingleLabelRuleEvaluation(const T& labelIndices, const IHeuristic& heuristic)
                : indexVector_(PartialIndexVector(1)), scoreVector_(DenseScoreVector<PartialIndexVector>(indexVector_)),
                  heuristic_(heuristic) {

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
                // TODO Implement
                return scoreVector_;
            }

    };

    LabelWiseSingleLabelRuleEvaluationFactory::LabelWiseSingleLabelRuleEvaluationFactory(
            std::unique_ptr<IHeuristic> heuristicPtr)
        : heuristicPtr_(std::move(heuristicPtr)) {

    }

    std::unique_ptr<IRuleEvaluation> LabelWiseSingleLabelRuleEvaluationFactory::create(
            const CompleteIndexVector& indexVector) const {
        return std::make_unique<LabelWiseSingleLabelRuleEvaluation<CompleteIndexVector>>(indexVector, *heuristicPtr_);
    }

    std::unique_ptr<IRuleEvaluation> LabelWiseSingleLabelRuleEvaluationFactory::create(
            const PartialIndexVector& indexVector) const {
        return std::make_unique<LabelWiseSingleLabelRuleEvaluation<PartialIndexVector>>(indexVector, *heuristicPtr_);
    }

}
