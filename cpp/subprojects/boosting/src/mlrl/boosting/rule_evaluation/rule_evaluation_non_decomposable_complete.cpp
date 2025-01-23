#include "mlrl/boosting/rule_evaluation/rule_evaluation_non_decomposable_complete.hpp"

#include "rule_evaluation_non_decomposable_complete_common.hpp"

namespace boosting {

    NonDecomposableCompleteRuleEvaluationFactory::NonDecomposableCompleteRuleEvaluationFactory(
      float32 l1RegularizationWeight, float32 l2RegularizationWeight, const BlasFactory& blasFactory,
      const LapackFactory& lapackFactory)
        : l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight),
          blasFactory_(blasFactory), lapackFactory_(lapackFactory) {}

    std::unique_ptr<IRuleEvaluation<DenseNonDecomposableStatisticVector<float64>>>
      NonDecomposableCompleteRuleEvaluationFactory::create(
        const DenseNonDecomposableStatisticVector<float64>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        return std::make_unique<DenseNonDecomposableCompleteRuleEvaluation<DenseNonDecomposableStatisticVector<float64>,
                                                                           CompleteIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_, blasFactory_.create64Bit(),
          lapackFactory_.create64Bit());
    }

    std::unique_ptr<IRuleEvaluation<DenseNonDecomposableStatisticVector<float64>>>
      NonDecomposableCompleteRuleEvaluationFactory::create(
        const DenseNonDecomposableStatisticVector<float64>& statisticVector,
        const PartialIndexVector& indexVector) const {
        return std::make_unique<
          DenseNonDecomposableCompleteRuleEvaluation<DenseNonDecomposableStatisticVector<float64>, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_, blasFactory_.create64Bit(),
          lapackFactory_.create64Bit());
    }

}
