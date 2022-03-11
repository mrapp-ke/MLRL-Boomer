#include "boosting/rule_evaluation/rule_evaluation_label_wise_partial_fixed.hpp"


namespace boosting {

    LabelWiseFixedPartialRuleEvaluationFactory::LabelWiseFixedPartialRuleEvaluationFactory(
            float32 labelRatio, uint32 minLabels, uint32 maxLabels, float64 l1RegularizationWeight,
            float64 l2RegularizationWeight)
        : labelRatio_(labelRatio), minLabels_(minLabels), maxLabels_(maxLabels),
          l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight) {

    }

    std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> LabelWiseFixedPartialRuleEvaluationFactory::create(
            const DenseLabelWiseStatisticVector& statisticVector, const CompleteIndexVector& indexVector) const {
        // TODO
        return nullptr;
    }

    std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> LabelWiseFixedPartialRuleEvaluationFactory::create(
            const DenseLabelWiseStatisticVector& statisticVector, const PartialIndexVector& indexVector) const {
        // TODO
        return nullptr;
    }

}
