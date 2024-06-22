/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/seco/heuristics/heuristic.hpp"
#include "mlrl/seco/lift_functions/lift_function.hpp"
#include "mlrl/seco/rule_evaluation/rule_evaluation_decomposable.hpp"

#include <memory>

#include <memory>

namespace seco {

    /**
     * Allows to create instances of the class `IDecomposableRuleEvaluationFactory` that allow to calculate the
     * predictions of partial rules, which predict for a subset of the available labels.
     */
    class DecomposablePartialRuleEvaluationFactory final : public IDecomposableRuleEvaluationFactory {
        private:

            const std::unique_ptr<IHeuristicFactory> heuristicFactoryPtr_;

            const std::unique_ptr<ILiftFunctionFactory> liftFunctionFactoryPtr_;

        public:

            /**
             * @param heuristicFactoryPtr       An unique pointer to an object of type `IHeuristicFactory`, that allows
             *                                  to create implementations of the heuristic to be optimized
             * @param liftFunctionFactoryPtr    An unique pointer to an object of type `ILiftFunction` that should
             *                                  affect the quality of rules, depending on how many labels they predict
             */
            DecomposablePartialRuleEvaluationFactory(std::unique_ptr<IHeuristicFactory> heuristicFactoryPtr,
                                                     std::unique_ptr<ILiftFunctionFactory> liftFunctionFactoryPtr);

            std::unique_ptr<IRuleEvaluation> create(const CompleteIndexVector& indexVector) const override;

            std::unique_ptr<IRuleEvaluation> create(const PartialIndexVector& indexVector) const override;
    };

}
