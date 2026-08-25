#include "mlrl/boosting/rule_evaluation/rule_evaluation_decomposable_complete_binned.hpp"

#include "mlrl/boosting/rule_evaluation/simd/vector_math_decomposable_simd.hpp"
#include "mlrl/boosting/rule_evaluation/vector_math_decomposable.hpp"
#include "rule_evaluation_decomposable_binned_common.hpp"

namespace boosting {

    template<typename VectorMath>
    DecomposableCompleteBinnedRuleEvaluationFactory<VectorMath>::DecomposableCompleteBinnedRuleEvaluationFactory(
      float32 l1RegularizationWeight, float32 l2RegularizationWeight,
      std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr)
        : l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight),
          labelBinningFactoryPtr_(std::move(labelBinningFactoryPtr)) {}

    template<typename VectorMath>
    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVectorView<float32>>>
      DecomposableCompleteBinnedRuleEvaluationFactory<VectorMath>::create(
        const DenseDecomposableStatisticVectorView<float32>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning<float32>> labelBinningPtr = labelBinningFactoryPtr_->create32Bit();
        return std::make_unique<DecomposableCompleteBinnedRuleEvaluation<DenseDecomposableStatisticVectorView<float32>,
                                                                         CompleteIndexVector, VectorMath>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr));
    }

    template<typename VectorMath>
    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVectorView<float32>>>
      DecomposableCompleteBinnedRuleEvaluationFactory<VectorMath>::create(
        const DenseDecomposableStatisticVectorView<float32>& statisticVector,
        const PartialIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning<float32>> labelBinningPtr = labelBinningFactoryPtr_->create32Bit();
        return std::make_unique<DecomposableCompleteBinnedRuleEvaluation<DenseDecomposableStatisticVectorView<float32>,
                                                                         PartialIndexVector, VectorMath>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr));
    }

    template<typename VectorMath>
    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVectorView<float64>>>
      DecomposableCompleteBinnedRuleEvaluationFactory<VectorMath>::create(
        const DenseDecomposableStatisticVectorView<float64>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning<float64>> labelBinningPtr = labelBinningFactoryPtr_->create64Bit();
        return std::make_unique<DecomposableCompleteBinnedRuleEvaluation<DenseDecomposableStatisticVectorView<float64>,
                                                                         CompleteIndexVector, VectorMath>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr));
    }

    template<typename VectorMath>
    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVectorView<float64>>>
      DecomposableCompleteBinnedRuleEvaluationFactory<VectorMath>::create(
        const DenseDecomposableStatisticVectorView<float64>& statisticVector,
        const PartialIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning<float64>> labelBinningPtr = labelBinningFactoryPtr_->create64Bit();
        return std::make_unique<DecomposableCompleteBinnedRuleEvaluation<DenseDecomposableStatisticVectorView<float64>,
                                                                         PartialIndexVector, VectorMath>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr));
    }

    template class DecomposableCompleteBinnedRuleEvaluationFactory<SequentialDecomposableVectorMath>;
#if SIMD_SUPPORT_ENABLED
    template class DecomposableCompleteBinnedRuleEvaluationFactory<SimdDecomposableVectorMath>;
#endif

}
