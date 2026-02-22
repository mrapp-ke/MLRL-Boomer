#include "mlrl/boosting/rule_evaluation/rule_evaluation_decomposable_complete_binned.hpp"

#include "rule_evaluation_decomposable_binned_common.hpp"

namespace boosting {

    DecomposableCompleteBinnedRuleEvaluationFactory::DecomposableCompleteBinnedRuleEvaluationFactory(
      float32 l1RegularizationWeight, float32 l2RegularizationWeight,
      std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr)
        : l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight),
          labelBinningFactoryPtr_(std::move(labelBinningFactoryPtr)) {}

    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVectorView<float32>>>
      DecomposableCompleteBinnedRuleEvaluationFactory::create(
        const DenseDecomposableStatisticVectorView<float32>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning<float32>> labelBinningPtr = labelBinningFactoryPtr_->create32Bit();
        return std::make_unique<
          DecomposableCompleteBinnedRuleEvaluation<DenseDecomposableStatisticVectorView<float32>, CompleteIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr));
    }

    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVectorView<float32>>>
      DecomposableCompleteBinnedRuleEvaluationFactory::create(
        const DenseDecomposableStatisticVectorView<float32>& statisticVector,
        const PartialIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning<float32>> labelBinningPtr = labelBinningFactoryPtr_->create32Bit();
        return std::make_unique<
          DecomposableCompleteBinnedRuleEvaluation<DenseDecomposableStatisticVectorView<float32>, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr));
    }

    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVectorView<float64>>>
      DecomposableCompleteBinnedRuleEvaluationFactory::create(
        const DenseDecomposableStatisticVectorView<float64>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning<float64>> labelBinningPtr = labelBinningFactoryPtr_->create64Bit();
        return std::make_unique<
          DecomposableCompleteBinnedRuleEvaluation<DenseDecomposableStatisticVectorView<float64>, CompleteIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr));
    }

    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVectorView<float64>>>
      DecomposableCompleteBinnedRuleEvaluationFactory::create(
        const DenseDecomposableStatisticVectorView<float64>& statisticVector,
        const PartialIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning<float64>> labelBinningPtr = labelBinningFactoryPtr_->create64Bit();
        return std::make_unique<
          DecomposableCompleteBinnedRuleEvaluation<DenseDecomposableStatisticVectorView<float64>, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr));
    }

}
