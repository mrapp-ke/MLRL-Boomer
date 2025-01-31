/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/binning/label_binning.hpp"
#include "mlrl/boosting/rule_evaluation/rule_evaluation_decomposable.hpp"

#include <memory>

namespace boosting {

    /**
     * Allows to create instances of the class `IDecomposableRuleEvaluationFactory` that allow to calculate the
     * predictions of complete rules, which predict for all available outputs, using gradient-based label binning.
     */
    class DecomposableCompleteBinnedRuleEvaluationFactory final : public IDecomposableRuleEvaluationFactory {
        private:

            const float32 l1RegularizationWeight_;

            const float32 l2RegularizationWeight_;

            const std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr_;

        public:

            /**
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param labelBinningFactoryPtr    An unique pointer to an object of type `ILabelBinningFactory` that
             *                                  allows to create the implementation to be used to assign labels to bins
             */
            DecomposableCompleteBinnedRuleEvaluationFactory(
              float32 l1RegularizationWeight, float32 l2RegularizationWeight,
              std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr);

            std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVector<float64>>> create(
              const DenseDecomposableStatisticVector<float64>& statisticVector,
              const CompleteIndexVector& indexVector) const override;

            std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVector<float64>>> create(
              const DenseDecomposableStatisticVector<float64>& statisticVector,
              const PartialIndexVector& indexVector) const override;
    };

}
