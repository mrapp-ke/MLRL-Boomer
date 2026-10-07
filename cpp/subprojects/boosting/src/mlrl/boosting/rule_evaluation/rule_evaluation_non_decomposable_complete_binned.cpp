#include "mlrl/boosting/rule_evaluation/rule_evaluation_non_decomposable_complete_binned.hpp"

#include "mlrl/common/simd/memory.hpp"
#include "rule_evaluation_non_decomposable_binned_common.hpp"

namespace boosting {

    template<typename MemoryAllocator>
    NonDecomposableCompleteBinnedRuleEvaluationFactory<MemoryAllocator>::
      NonDecomposableCompleteBinnedRuleEvaluationFactory(float32 l1RegularizationWeight, float32 l2RegularizationWeight,
                                                         std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr,
                                                         const BlasFactory& blasFactory,
                                                         const LapackFactory& lapackFactory)
        : l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight),
          labelBinningFactoryPtr_(std::move(labelBinningFactoryPtr)), blasFactory_(blasFactory),
          lapackFactory_(lapackFactory) {}

    template<typename MemoryAllocator>
    std::unique_ptr<IRuleEvaluation<DenseNonDecomposableStatisticVectorView<float32>>>
      NonDecomposableCompleteBinnedRuleEvaluationFactory<MemoryAllocator>::create(
        const DenseNonDecomposableStatisticVectorView<float32>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning<float32>> labelBinningPtr = labelBinningFactoryPtr_->create32Bit();
        uint32 maxBins = labelBinningPtr->getMaxBins(indexVector.getNumElements());
        return std::make_unique<DenseNonDecomposableCompleteBinnedRuleEvaluation<
          DenseNonDecomposableStatisticVectorView<float32>, CompleteIndexVector, MemoryAllocator>>(
          indexVector, maxBins, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr),
          blasFactory_.create32Bit(), lapackFactory_.create32Bit());
    }

    template<typename MemoryAllocator>
    std::unique_ptr<IRuleEvaluation<DenseNonDecomposableStatisticVectorView<float32>>>
      NonDecomposableCompleteBinnedRuleEvaluationFactory<MemoryAllocator>::create(
        const DenseNonDecomposableStatisticVectorView<float32>& statisticVector,
        const PartialIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning<float32>> labelBinningPtr = labelBinningFactoryPtr_->create32Bit();
        uint32 maxBins = labelBinningPtr->getMaxBins(indexVector.getNumElements());
        return std::make_unique<DenseNonDecomposableCompleteBinnedRuleEvaluation<
          DenseNonDecomposableStatisticVectorView<float32>, PartialIndexVector, MemoryAllocator>>(
          indexVector, maxBins, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr),
          blasFactory_.create32Bit(), lapackFactory_.create32Bit());
    }

    template<typename MemoryAllocator>
    std::unique_ptr<IRuleEvaluation<DenseNonDecomposableStatisticVectorView<float64>>>
      NonDecomposableCompleteBinnedRuleEvaluationFactory<MemoryAllocator>::create(
        const DenseNonDecomposableStatisticVectorView<float64>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning<float64>> labelBinningPtr = labelBinningFactoryPtr_->create64Bit();
        uint32 maxBins = labelBinningPtr->getMaxBins(indexVector.getNumElements());
        return std::make_unique<DenseNonDecomposableCompleteBinnedRuleEvaluation<
          DenseNonDecomposableStatisticVectorView<float64>, CompleteIndexVector, MemoryAllocator>>(
          indexVector, maxBins, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr),
          blasFactory_.create64Bit(), lapackFactory_.create64Bit());
    }

    template<typename MemoryAllocator>
    std::unique_ptr<IRuleEvaluation<DenseNonDecomposableStatisticVectorView<float64>>>
      NonDecomposableCompleteBinnedRuleEvaluationFactory<MemoryAllocator>::create(
        const DenseNonDecomposableStatisticVectorView<float64>& statisticVector,
        const PartialIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning<float64>> labelBinningPtr = labelBinningFactoryPtr_->create64Bit();
        uint32 maxBins = labelBinningPtr->getMaxBins(indexVector.getNumElements());
        return std::make_unique<DenseNonDecomposableCompleteBinnedRuleEvaluation<
          DenseNonDecomposableStatisticVectorView<float64>, PartialIndexVector, MemoryAllocator>>(
          indexVector, maxBins, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr),
          blasFactory_.create64Bit(), lapackFactory_.create64Bit());
    }

    template class NonDecomposableCompleteBinnedRuleEvaluationFactory<DefaultMemoryAllocator>;

#if SIMD_SUPPORT_ENABLED
    template class NonDecomposableCompleteBinnedRuleEvaluationFactory<SimdMemoryAllocator>;
#endif
}
