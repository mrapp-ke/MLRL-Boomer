/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/seco/heuristics/heuristic.hpp"
#include "mlrl/seco/rule_evaluation/rule_evaluation_decomposable.hpp"

#include <memory>

namespace seco {

    /**
     * Allows to create instances of the class `IDecomposableRuleEvaluationFactory` that allow to calculate the
     * predictions of single-output rules, which predict for a single output.
     */
    class DecomposableSingleOutputRuleEvaluationFactory final : public IDecomposableRuleEvaluationFactory {
        private:

            const std::unique_ptr<IHeuristicFactory> heuristicFactoryPtr_;

        public:

            /**
             * @param heuristicFactoryPtr An unique pointer to an object of type `IHeuristicFactory`, that allows to
             *                            create implementations of the heuristic to be optimized
             */
            DecomposableSingleOutputRuleEvaluationFactory(std::unique_ptr<IHeuristicFactory> heuristicFactoryPtr);

            std::unique_ptr<IRuleEvaluation<DenseConfusionMatrixVector<uint32>>> create(
              const DenseConfusionMatrixVector<uint32>& statisticVector,
              const CompleteIndexVector& indexVector) const override;

            std::unique_ptr<IRuleEvaluation<DenseConfusionMatrixVector<uint32>>> create(
              const DenseConfusionMatrixVector<uint32>& statisticVector,
              const PartialIndexVector& indexVector) const override;

            std::unique_ptr<IRuleEvaluation<DenseConfusionMatrixVector<float32>>> create(
              const DenseConfusionMatrixVector<float32>& statisticVector,
              const CompleteIndexVector& indexVector) const override;

            std::unique_ptr<IRuleEvaluation<DenseConfusionMatrixVector<float32>>> create(
              const DenseConfusionMatrixVector<float32>& statisticVector,
              const PartialIndexVector& indexVector) const override;
    };

}
