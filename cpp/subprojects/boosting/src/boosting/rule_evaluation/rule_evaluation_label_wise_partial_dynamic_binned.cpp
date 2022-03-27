#include "boosting/rule_evaluation/rule_evaluation_label_wise_partial_dynamic_binned.hpp"
#include "rule_evaluation_label_wise_binned_common.hpp"


namespace boosting {

    LabelWiseDynamicPartialBinnedRuleEvaluationFactory::LabelWiseDynamicPartialBinnedRuleEvaluationFactory(
            float32 threshold, float64 l1RegularizationWeight, float64 l2RegularizationWeight,
            std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr)
        : threshold_(threshold), l1RegularizationWeight_(l1RegularizationWeight),
          l2RegularizationWeight_(l2RegularizationWeight), labelBinningFactoryPtr_(std::move(labelBinningFactoryPtr)) {

    }

    std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> LabelWiseDynamicPartialBinnedRuleEvaluationFactory::create(
            const DenseLabelWiseStatisticVector& statisticVector, const CompleteIndexVector& indexVector) const {
        // TODO
        return nullptr;
    }

    std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> LabelWiseDynamicPartialBinnedRuleEvaluationFactory::create(
            const DenseLabelWiseStatisticVector& statisticVector, const PartialIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning> labelBinningPtr = labelBinningFactoryPtr_->create();
        return std::make_unique<DenseLabelWiseCompleteBinnedRuleEvaluation<PartialIndexVector>>(
            indexVector, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr));
    }

}
