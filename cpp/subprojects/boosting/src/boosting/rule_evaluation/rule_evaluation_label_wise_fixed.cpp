#include "boosting/rule_evaluation/rule_evaluation_label_wise_fixed.hpp"


namespace boosting {

    LabelWiseFixedRuleEvaluationFactory::LabelWiseFixedRuleEvaluationFactory(float64 l1RegularizationWeight,
                                                                             float64 l2RegularizationWeight)
        : l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight) {

    }

    std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> LabelWiseFixedRuleEvaluationFactory::create(
            const DenseLabelWiseStatisticVector& statisticVector, const CompleteIndexVector& indexVector) const {
        // TODO
        return nullptr;
    }

    std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> LabelWiseFixedRuleEvaluationFactory::create(
            const DenseLabelWiseStatisticVector& statisticVector, const PartialIndexVector& indexVector) const {
        // TODO
        return nullptr;
    }

}
