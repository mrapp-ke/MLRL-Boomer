#include "mlrl/boosting/rule_evaluation/rule_evaluation_decomposable_complete.hpp"

#include "rule_evaluation_decomposable_complete_common.hpp"

namespace boosting {

    DecomposableCompleteRuleEvaluationFactory::DecomposableCompleteRuleEvaluationFactory(float64 l1RegularizationWeight,
                                                                                         float64 l2RegularizationWeight)
        : l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight) {}

    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVector<float64>>>
      DecomposableCompleteRuleEvaluationFactory::create(
        const DenseDecomposableStatisticVector<float64>& statisticVector,
        const CompleteIndexVector& indexVector) const {
        return std::make_unique<
          DecomposableCompleteRuleEvaluation<DenseDecomposableStatisticVector<float64>, CompleteIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<DenseDecomposableStatisticVector<float64>>>
      DecomposableCompleteRuleEvaluationFactory::create(
        const DenseDecomposableStatisticVector<float64>& statisticVector, const PartialIndexVector& indexVector) const {
        return std::make_unique<
          DecomposableCompleteRuleEvaluation<DenseDecomposableStatisticVector<float64>, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

}
