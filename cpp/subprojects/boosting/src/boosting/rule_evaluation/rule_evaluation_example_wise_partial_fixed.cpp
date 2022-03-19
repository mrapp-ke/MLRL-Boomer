#include "boosting/rule_evaluation/rule_evaluation_example_wise_partial_fixed.hpp"


namespace boosting {

    ExampleWiseFixedPartialRuleEvaluationFactory::ExampleWiseFixedPartialRuleEvaluationFactory(
            float32 labelRatio, uint32 minLabels, uint32 maxLabels, float64 l1RegularizationWeight,
            float64 l2RegularizationWeight, const Blas& blas, const Lapack& lapack)
        : labelRatio_(labelRatio), minLabels_(minLabels), maxLabels_(maxLabels),
          l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight), blas_(blas),
          lapack_(lapack) {

    }

    std::unique_ptr<IRuleEvaluation<DenseExampleWiseStatisticVector>> ExampleWiseFixedPartialRuleEvaluationFactory::create(
            const DenseExampleWiseStatisticVector& statisticVector, const CompleteIndexVector& indexVector) const {
        // TODO
        return nullptr;
    }

    std::unique_ptr<IRuleEvaluation<DenseExampleWiseStatisticVector>> ExampleWiseFixedPartialRuleEvaluationFactory::create(
            const DenseExampleWiseStatisticVector& statisticVector, const PartialIndexVector& indexVector) const {
        // TODO
        return nullptr;
    }

}
