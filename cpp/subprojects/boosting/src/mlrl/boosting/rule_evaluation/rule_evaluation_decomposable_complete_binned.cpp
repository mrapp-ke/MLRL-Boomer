#include "mlrl/boosting/rule_evaluation/rule_evaluation_decomposable_complete_binned.hpp"

#include "rule_evaluation_decomposable_binned_common.hpp"

namespace boosting {

    DecomposableCompleteBinnedRuleEvaluationFactory::DecomposableCompleteBinnedRuleEvaluationFactory(
      float32 l1RegularizationWeight, float32 l2RegularizationWeight,
      std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr)
        : l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight),
          labelBinningFactoryPtr_(std::move(labelBinningFactoryPtr)) {}

    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVector<float64>>>
      DecomposableCompleteBinnedRuleEvaluationFactory::create(
        const DenseDecomposableStatisticVector<float64>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning> labelBinningPtr = labelBinningFactoryPtr_->create();
        return std::make_unique<
          DecomposableCompleteBinnedRuleEvaluation<DenseDecomposableStatisticVector<float64>, CompleteIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr));
    }

    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVector<float64>>>
      DecomposableCompleteBinnedRuleEvaluationFactory::create(
        const DenseDecomposableStatisticVector<float64>& statisticVector, const PartialIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning> labelBinningPtr = labelBinningFactoryPtr_->create();
        return std::make_unique<
          DecomposableCompleteBinnedRuleEvaluation<DenseDecomposableStatisticVector<float64>, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr));
    }

}
