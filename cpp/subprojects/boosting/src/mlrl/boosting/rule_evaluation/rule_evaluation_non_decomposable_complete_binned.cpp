#include "mlrl/boosting/rule_evaluation/rule_evaluation_non_decomposable_complete_binned.hpp"

#include "rule_evaluation_non_decomposable_binned_common.hpp"

namespace boosting {

    NonDecomposableCompleteBinnedRuleEvaluationFactory::NonDecomposableCompleteBinnedRuleEvaluationFactory(
      float64 l1RegularizationWeight, float64 l2RegularizationWeight,
      std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr, const Blas& blas, const Lapack& lapack)
        : l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight),
          labelBinningFactoryPtr_(std::move(labelBinningFactoryPtr)), blas_(blas), lapack_(lapack) {}

    std::unique_ptr<IRuleEvaluation<DenseNonDecomposableStatisticVector<float64>>>
      NonDecomposableCompleteBinnedRuleEvaluationFactory::create(
        const DenseNonDecomposableStatisticVector<float64>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning> labelBinningPtr = labelBinningFactoryPtr_->create();
        uint32 maxBins = labelBinningPtr->getMaxBins(indexVector.getNumElements());
        return std::make_unique<DenseNonDecomposableCompleteBinnedRuleEvaluation<CompleteIndexVector>>(
          indexVector, maxBins, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr), blas_,
          lapack_);
    }

    std::unique_ptr<IRuleEvaluation<DenseNonDecomposableStatisticVector<float64>>>
      NonDecomposableCompleteBinnedRuleEvaluationFactory::create(
        const DenseNonDecomposableStatisticVector<float64>& statisticVector,
        const PartialIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning> labelBinningPtr = labelBinningFactoryPtr_->create();
        uint32 maxBins = labelBinningPtr->getMaxBins(indexVector.getNumElements());
        return std::make_unique<DenseNonDecomposableCompleteBinnedRuleEvaluation<PartialIndexVector>>(
          indexVector, maxBins, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr), blas_,
          lapack_);
    }

}
