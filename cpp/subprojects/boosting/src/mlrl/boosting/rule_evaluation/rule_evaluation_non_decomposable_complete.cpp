#include "mlrl/boosting/rule_evaluation/rule_evaluation_non_decomposable_complete.hpp"

#include "rule_evaluation_non_decomposable_complete_common.hpp"

namespace boosting {

    NonDecomposableCompleteRuleEvaluationFactory::NonDecomposableCompleteRuleEvaluationFactory(
      float64 l1RegularizationWeight, float64 l2RegularizationWeight, const Blas& blas, const Lapack& lapack)
        : l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight), blas_(blas),
          lapack_(lapack) {}

    std::unique_ptr<IRuleEvaluation<DenseNonDecomposableStatisticVector>>
      NonDecomposableCompleteRuleEvaluationFactory::create(const DenseNonDecomposableStatisticVector& statisticVector,
                                                           const CompleteIndexVector& indexVector) const {
        return std::make_unique<DenseNonDecomposableCompleteRuleEvaluation<CompleteIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_, blas_, lapack_);
    }

    std::unique_ptr<IRuleEvaluation<DenseNonDecomposableStatisticVector>>
      NonDecomposableCompleteRuleEvaluationFactory::create(const DenseNonDecomposableStatisticVector& statisticVector,
                                                           const PartialIndexVector& indexVector) const {
        return std::make_unique<DenseNonDecomposableCompleteRuleEvaluation<PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_, blas_, lapack_);
    }

}
