#include "seco/rule_evaluation/rule_evaluation_label_wise_partial.hpp"
#include "common/validation.hpp"


namespace seco {

    /**
     * Allows to calculate the predictions of partial rules, as well as corresponding quality scores, such that they
     * optimize a heuristic that is applied using label-wise averaging.
     *
     * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
     */
    template<typename T>
    class LabelWisePartialRuleEvaluation final : public IRuleEvaluation {

        private:

            const IHeuristic& heuristic_;

            const ILiftFunction& liftFunction_;

        public:

            /**
             * @param labelIndices  A reference to an object of template type `T` that provides access to the indices of
             *                      the labels for which the rules may predict
             * @param heuristic     A reference to an object of type `IHeuristic`, implementing the heuristic to be
             *                      optimized
             * @param liftFunction  A reference to an object of type `ILiftFunction` that should affect the quality
             *                      scores of rules, depending on how many labels they predict
             */
            LabelWisePartialRuleEvaluation(const T& labelIndices, const IHeuristic& heuristic,
                                           const ILiftFunction& liftFunction)
                : heuristic_(heuristic), liftFunction_(liftFunction) {

            }

            const IScoreVector& calculatePrediction(
                    const BinarySparseArrayVector& majorityLabelVector,
                    const DenseConfusionMatrixVector& confusionMatricesTotal,
                    const DenseConfusionMatrixVector& confusionMatricesCovered) override {
                // TODO Implement
            }

    };

    LabelWisePartialRuleEvaluationFactory::LabelWisePartialRuleEvaluationFactory(
            std::unique_ptr<IHeuristic> heuristicPtr, std::unique_ptr<ILiftFunction> liftFunctionPtr)
        : heuristicPtr_(std::move(heuristicPtr)), liftFunctionPtr_(std::move(liftFunctionPtr)) {
        assertNotNull("heuristicPtr", heuristicPtr_.get());
        assertNotNull("liftFunctionPtr", liftFunctionPtr_.get());
    }

    std::unique_ptr<IRuleEvaluation> LabelWisePartialRuleEvaluationFactory::create(
            const CompleteIndexVector& indexVector) const {
        return std::make_unique<LabelWisePartialRuleEvaluation<CompleteIndexVector>>(indexVector, *heuristicPtr_,
                                                                                     *liftFunctionPtr_);
    }

    std::unique_ptr<IRuleEvaluation> LabelWisePartialRuleEvaluationFactory::create(
            const PartialIndexVector& indexVector) const {
        return std::make_unique<LabelWisePartialRuleEvaluation<PartialIndexVector>>(indexVector, *heuristicPtr_,
                                                                                    *liftFunctionPtr_);
    }

}
