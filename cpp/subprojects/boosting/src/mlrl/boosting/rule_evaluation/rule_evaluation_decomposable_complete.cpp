#include "mlrl/boosting/rule_evaluation/rule_evaluation_decomposable_complete.hpp"

#include "mlrl/boosting/rule_evaluation/simd/vector_math_decomposable_simd.hpp"
#include "mlrl/boosting/rule_evaluation/vector_math_decomposable.hpp"
#include "mlrl/common/simd/memory.hpp"
#include "rule_evaluation_decomposable_complete_common.hpp"

namespace boosting {

    template<typename VectorMath, typename MemoryAllocator>
    DecomposableCompleteRuleEvaluationFactory<VectorMath, MemoryAllocator>::DecomposableCompleteRuleEvaluationFactory(
      float32 l1RegularizationWeight, float32 l2RegularizationWeight)
        : l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight) {}

    template<typename VectorMath, typename MemoryAllocator>
    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVectorView<float32>>>
      DecomposableCompleteRuleEvaluationFactory<VectorMath, MemoryAllocator>::create(
        const DenseDecomposableStatisticVectorView<float32>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        return std::make_unique<DecomposableCompleteRuleEvaluation<DenseDecomposableStatisticVectorView<float32>,
                                                                   CompleteIndexVector, VectorMath, MemoryAllocator>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    template<typename VectorMath, typename MemoryAllocator>
    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVectorView<float32>>>
      DecomposableCompleteRuleEvaluationFactory<VectorMath, MemoryAllocator>::create(
        const DenseDecomposableStatisticVectorView<float32>& statisticVector,
        const PartialIndexVector& indexVector) const {
        return std::make_unique<DecomposableCompleteRuleEvaluation<DenseDecomposableStatisticVectorView<float32>,
                                                                   PartialIndexVector, VectorMath, MemoryAllocator>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    template<typename VectorMath, typename MemoryAllocator>
    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVectorView<float64>>>
      DecomposableCompleteRuleEvaluationFactory<VectorMath, MemoryAllocator>::create(
        const DenseDecomposableStatisticVectorView<float64>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        return std::make_unique<DecomposableCompleteRuleEvaluation<DenseDecomposableStatisticVectorView<float64>,
                                                                   CompleteIndexVector, VectorMath, MemoryAllocator>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    template<typename VectorMath, typename MemoryAllocator>
    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVectorView<float64>>>
      DecomposableCompleteRuleEvaluationFactory<VectorMath, MemoryAllocator>::create(
        const DenseDecomposableStatisticVectorView<float64>& statisticVector,
        const PartialIndexVector& indexVector) const {
        return std::make_unique<DecomposableCompleteRuleEvaluation<DenseDecomposableStatisticVectorView<float64>,
                                                                   PartialIndexVector, VectorMath, MemoryAllocator>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    template class DecomposableCompleteRuleEvaluationFactory<SequentialDecomposableVectorMath, DefaultMemoryAllocator>;
#if SIMD_SUPPORT_ENABLED
    template class DecomposableCompleteRuleEvaluationFactory<SimdDecomposableVectorMath, SimdMemoryAllocator>;
#endif
}
