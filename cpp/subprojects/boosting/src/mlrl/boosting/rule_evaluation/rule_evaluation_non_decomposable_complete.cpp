#include "mlrl/boosting/rule_evaluation/rule_evaluation_non_decomposable_complete.hpp"

#include "rule_evaluation_non_decomposable_complete_common.hpp"

namespace boosting {

    template<typename MemoryAllocator>
    NonDecomposableCompleteRuleEvaluationFactory<MemoryAllocator>::NonDecomposableCompleteRuleEvaluationFactory(
      float32 l1RegularizationWeight, float32 l2RegularizationWeight, const BlasFactory& blasFactory,
      const LapackFactory& lapackFactory)
        : l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight),
          blasFactory_(blasFactory), lapackFactory_(lapackFactory) {}

    template<typename MemoryAllocator>
    std::unique_ptr<IRuleEvaluation<DenseNonDecomposableStatisticVectorView<float32>>>
      NonDecomposableCompleteRuleEvaluationFactory<MemoryAllocator>::create(
        const DenseNonDecomposableStatisticVectorView<float32>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        return std::make_unique<DenseNonDecomposableCompleteRuleEvaluation<
          DenseNonDecomposableStatisticVectorView<float32>, CompleteIndexVector, MemoryAllocator>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_, blasFactory_.create32Bit(),
          lapackFactory_.create32Bit());
    }

    template<typename MemoryAllocator>
    std::unique_ptr<IRuleEvaluation<DenseNonDecomposableStatisticVectorView<float32>>>
      NonDecomposableCompleteRuleEvaluationFactory<MemoryAllocator>::create(
        const DenseNonDecomposableStatisticVectorView<float32>& statisticVector,
        const PartialIndexVector& indexVector) const {
        return std::make_unique<DenseNonDecomposableCompleteRuleEvaluation<
          DenseNonDecomposableStatisticVectorView<float32>, PartialIndexVector, MemoryAllocator>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_, blasFactory_.create32Bit(),
          lapackFactory_.create32Bit());
    }

    template<typename MemoryAllocator>
    std::unique_ptr<IRuleEvaluation<DenseNonDecomposableStatisticVectorView<float64>>>
      NonDecomposableCompleteRuleEvaluationFactory<MemoryAllocator>::create(
        const DenseNonDecomposableStatisticVectorView<float64>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        return std::make_unique<DenseNonDecomposableCompleteRuleEvaluation<
          DenseNonDecomposableStatisticVectorView<float64>, CompleteIndexVector, MemoryAllocator>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_, blasFactory_.create64Bit(),
          lapackFactory_.create64Bit());
    }

    template<typename MemoryAllocator>
    std::unique_ptr<IRuleEvaluation<DenseNonDecomposableStatisticVectorView<float64>>>
      NonDecomposableCompleteRuleEvaluationFactory<MemoryAllocator>::create(
        const DenseNonDecomposableStatisticVectorView<float64>& statisticVector,
        const PartialIndexVector& indexVector) const {
        return std::make_unique<DenseNonDecomposableCompleteRuleEvaluation<
          DenseNonDecomposableStatisticVectorView<float64>, PartialIndexVector, MemoryAllocator>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_, blasFactory_.create64Bit(),
          lapackFactory_.create64Bit());
    }

    template class NonDecomposableCompleteRuleEvaluationFactory<DefaultMemoryAllocator>;

#if SIMD_SUPPORT_ENABLED
    template class NonDecomposableCompleteRuleEvaluationFactory<SimdMemoryAllocator>;
#endif
}
