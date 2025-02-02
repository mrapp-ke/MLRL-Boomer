#include "mlrl/boosting/rule_evaluation/rule_evaluation_non_decomposable_complete_binned.hpp"

#include "rule_evaluation_non_decomposable_binned_common.hpp"

namespace boosting {

    NonDecomposableCompleteBinnedRuleEvaluationFactory::NonDecomposableCompleteBinnedRuleEvaluationFactory(
      float32 l1RegularizationWeight, float32 l2RegularizationWeight,
      std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr, const BlasFactory& blasFactory,
      const LapackFactory& lapackFactory)
        : l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight),
          labelBinningFactoryPtr_(std::move(labelBinningFactoryPtr)), blasFactory_(blasFactory),
          lapackFactory_(lapackFactory) {}

    std::unique_ptr<IRuleEvaluation<DenseNonDecomposableStatisticVector<float64>>>
      NonDecomposableCompleteBinnedRuleEvaluationFactory::create(
        const DenseNonDecomposableStatisticVector<float64>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning<float64>> labelBinningPtr = labelBinningFactoryPtr_->create();
        uint32 maxBins = labelBinningPtr->getMaxBins(indexVector.getNumElements());
        return std::make_unique<DenseNonDecomposableCompleteBinnedRuleEvaluation<
          DenseNonDecomposableStatisticVector<float64>, CompleteIndexVector>>(
          indexVector, maxBins, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr),
          blasFactory_.create64Bit(), lapackFactory_.create64Bit());
    }

    std::unique_ptr<IRuleEvaluation<DenseNonDecomposableStatisticVector<float64>>>
      NonDecomposableCompleteBinnedRuleEvaluationFactory::create(
        const DenseNonDecomposableStatisticVector<float64>& statisticVector,
        const PartialIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning<float64>> labelBinningPtr = labelBinningFactoryPtr_->create();
        uint32 maxBins = labelBinningPtr->getMaxBins(indexVector.getNumElements());
        return std::make_unique<DenseNonDecomposableCompleteBinnedRuleEvaluation<
          DenseNonDecomposableStatisticVector<float64>, PartialIndexVector>>(
          indexVector, maxBins, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr),
          blasFactory_.create64Bit(), lapackFactory_.create64Bit());
    }

}
