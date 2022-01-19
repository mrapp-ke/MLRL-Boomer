/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "seco/rule_evaluation/rule_evaluation_label_wise.hpp"
#include "seco/heuristics/heuristic.hpp"
#include "seco/lift_functions/lift_function.hpp"


namespace seco {

    /**
     * Allows to create instances of the class `LabelWisePartialRuleEvaluation`.
     */
    class LabelWisePartialRuleEvaluationFactory final : public ILabelWiseRuleEvaluationFactory {

        private:

            std::unique_ptr<IHeuristicFactory> heuristicFactoryPtr_;

            std::unique_ptr<ILiftFunctionFactory> liftFunctionFactoryPtr_;

        public:

            /**
             * @param heuristicFactoryPtr       An unique pointer to an object of type `IHeuristicFactory`, that allows
             *                                  to create implementations of the heuristic to be optimized
             * @param liftFunctionFactoryPtr    An unique pointer to an object of type `ILiftFunction` that should
             *                                  affect the quality scores of rules, depending on how many labels they
             *                                  predict
             */
            LabelWisePartialRuleEvaluationFactory(std::unique_ptr<IHeuristicFactory> heuristicFactoryPtr,
                                                  std::unique_ptr<ILiftFunctionFactory> liftFunctionFactoryPtr);

            std::unique_ptr<IRuleEvaluation> create(const CompleteIndexVector& indexVector) const override;

            std::unique_ptr<IRuleEvaluation> create(const PartialIndexVector& indexVector) const override;

    };

}
