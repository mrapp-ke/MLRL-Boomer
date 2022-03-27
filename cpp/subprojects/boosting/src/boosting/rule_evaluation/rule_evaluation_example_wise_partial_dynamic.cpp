#include "boosting/rule_evaluation/rule_evaluation_example_wise_partial_dynamic.hpp"
#include "rule_evaluation_example_wise_complete_common.hpp"


namespace boosting {

    ExampleWiseDynamicPartialRuleEvaluationFactory::ExampleWiseDynamicPartialRuleEvaluationFactory(
            float32 threshold, float64 l1RegularizationWeight, float64 l2RegularizationWeight, const Blas& blas,
            const Lapack& lapack)
        : threshold_(threshold), l1RegularizationWeight_(l1RegularizationWeight),
          l2RegularizationWeight_(l2RegularizationWeight), blas_(blas), lapack_(lapack) {

    }

    std::unique_ptr<IRuleEvaluation<DenseExampleWiseStatisticVector>> ExampleWiseDynamicPartialRuleEvaluationFactory::create(
            const DenseExampleWiseStatisticVector& statisticVector, const CompleteIndexVector& indexVector) const {
        // TODO
        return nullptr;
    }

    std::unique_ptr<IRuleEvaluation<DenseExampleWiseStatisticVector>> ExampleWiseDynamicPartialRuleEvaluationFactory::create(
            const DenseExampleWiseStatisticVector& statisticVector, const PartialIndexVector& indexVector) const {
        return std::make_unique<DenseExampleWiseCompleteRuleEvaluation<PartialIndexVector>>(
            indexVector, l1RegularizationWeight_, l2RegularizationWeight_, blas_, lapack_);;
    }

}
