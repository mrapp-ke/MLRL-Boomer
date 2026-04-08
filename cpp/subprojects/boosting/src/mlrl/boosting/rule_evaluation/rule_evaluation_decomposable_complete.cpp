#include "mlrl/boosting/rule_evaluation/rule_evaluation_decomposable_complete.hpp"

#include "mlrl/boosting/rule_evaluation/simd/vector_math_decomposable.hpp"
#include "mlrl/boosting/rule_evaluation/vector_math_decomposable.hpp"
#include "rule_evaluation_decomposable_complete_common.hpp"

namespace boosting {

    template<typename VectorMath>
    DecomposableCompleteRuleEvaluationFactory<VectorMath>::DecomposableCompleteRuleEvaluationFactory(
      float32 l1RegularizationWeight, float32 l2RegularizationWeight)
        : l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight) {}

    template<typename VectorMath>
    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVectorView<float32>>>
      DecomposableCompleteRuleEvaluationFactory<VectorMath>::create(
        const DenseDecomposableStatisticVectorView<float32>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        return std::make_unique<DecomposableCompleteRuleEvaluation<DenseDecomposableStatisticVectorView<float32>,
                                                                   CompleteIndexVector, VectorMath>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    template<typename VectorMath>
    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVectorView<float32>>>
      DecomposableCompleteRuleEvaluationFactory<VectorMath>::create(
        const DenseDecomposableStatisticVectorView<float32>& statisticVector,
        const PartialIndexVector& indexVector) const {
        return std::make_unique<DecomposableCompleteRuleEvaluation<DenseDecomposableStatisticVectorView<float32>,
                                                                   PartialIndexVector, VectorMath>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    template<typename VectorMath>
    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVectorView<float64>>>
      DecomposableCompleteRuleEvaluationFactory<VectorMath>::create(
        const DenseDecomposableStatisticVectorView<float64>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        return std::make_unique<DecomposableCompleteRuleEvaluation<DenseDecomposableStatisticVectorView<float64>,
                                                                   CompleteIndexVector, VectorMath>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    template<typename VectorMath>
    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVectorView<float64>>>
      DecomposableCompleteRuleEvaluationFactory<VectorMath>::create(
        const DenseDecomposableStatisticVectorView<float64>& statisticVector,
        const PartialIndexVector& indexVector) const {
        return std::make_unique<DecomposableCompleteRuleEvaluation<DenseDecomposableStatisticVectorView<float64>,
                                                                   PartialIndexVector, VectorMath>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    template class DecomposableCompleteRuleEvaluationFactory<SequentialDecomposableVectorMath>;
#if SIMD_SUPPORT_ENABLED
    template class DecomposableCompleteRuleEvaluationFactory<SimdDecomposableVectorMath>;
#endif
}
