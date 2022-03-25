#include "boosting/rule_evaluation/rule_evaluation_label_wise_partial_dynamic.hpp"
#include "rule_evaluation_label_wise_complete_common.hpp"


namespace boosting {

    LabelWiseDynamicPartialRuleEvaluationFactory::LabelWiseDynamicPartialRuleEvaluationFactory(
            float32 threshold, float64 l1RegularizationWeight, float64 l2RegularizationWeight)
        : threshold_(threshold), l1RegularizationWeight_(l1RegularizationWeight),
          l2RegularizationWeight_(l2RegularizationWeight) {

    }

    std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> LabelWiseDynamicPartialRuleEvaluationFactory::create(
            const DenseLabelWiseStatisticVector& statisticVector, const CompleteIndexVector& indexVector) const {
        // TODO
        return nullptr;
    }

    std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> LabelWiseDynamicPartialRuleEvaluationFactory::create(
            const DenseLabelWiseStatisticVector& statisticVector, const PartialIndexVector& indexVector) const {
        return std::make_unique<DenseLabelWiseCompleteRuleEvaluation<PartialIndexVector>>(indexVector,
                                                                                          l1RegularizationWeight_,
                                                                                          l2RegularizationWeight_);
    }

}
