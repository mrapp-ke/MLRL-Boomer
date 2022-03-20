#include "boosting/rule_evaluation/rule_evaluation_example_wise_partial_fixed_binned.hpp"


namespace boosting {

    ExampleWiseFixedPartialBinnedRuleEvaluationFactory::ExampleWiseFixedPartialBinnedRuleEvaluationFactory(
            float32 labelRatio, uint32 minLabels, uint32 maxLabels, float64 l1RegularizationWeight,
            float64 l2RegularizationWeight, std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr,
            const Blas& blas, const Lapack& lapack)
        : labelRatio_(labelRatio), minLabels_(minLabels), maxLabels_(maxLabels),
          l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight),
          labelBinningFactoryPtr_(std::move(labelBinningFactoryPtr)), blas_(blas), lapack_(lapack) {

    }

    std::unique_ptr<IRuleEvaluation<DenseExampleWiseStatisticVector>> ExampleWiseFixedPartialBinnedRuleEvaluationFactory::create(
            const DenseExampleWiseStatisticVector& statisticVector, const CompleteIndexVector& indexVector) const {
        // TODO
        return nullptr;
    }

    std::unique_ptr<IRuleEvaluation<DenseExampleWiseStatisticVector>> ExampleWiseFixedPartialBinnedRuleEvaluationFactory::create(
            const DenseExampleWiseStatisticVector& statisticVector, const PartialIndexVector& indexVector) const {
        // TODO
        return nullptr;
    }

}
