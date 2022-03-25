/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/rule_evaluation/rule_evaluation_label_wise.hpp"


namespace boosting {

    /**
     * Allows to create instances of the class `ILabelWiseRuleEvaluationFactory` that allow to calculate the predictions
     * of partial rules, which predict for a subset of the available labels that is determined dynamically.
     */
    class LabelWiseDynamicPartialRuleEvaluationFactory final : public ILabelWiseRuleEvaluationFactory {

        private:

            float32 threshold_;

            float64 l1RegularizationWeight_;

            float64 l2RegularizationWeight_;

        public:

            /**
             * @param threshold                 A threshold that affects for how many labels the rule heads should
             *                                  predict. A smaller threshold results in less labels being selected. A
             *                                  greater threshold results in more labels being selected. E.g., a
             *                                  threshold of 0.2 means that a rule will only predict for a label if the
             *                                  estimated predictive quality `q` for this particular label satisfies the
             *                                  inequality `q^2 > q_max^2 * (1 - 0.2)`, where `q_max` is the best
             *                                  quality among all labels. Must be in (0, 1)
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             */
            LabelWiseDynamicPartialRuleEvaluationFactory(float32 threshold, float64 l1RegularizationWeight,
                                                         float64 l2RegularizationWeight);

            std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> create(
                const DenseLabelWiseStatisticVector& statisticVector,
                const CompleteIndexVector& indexVector) const override;

            std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> create(
                const DenseLabelWiseStatisticVector& statisticVector,
                const PartialIndexVector& indexVector) const override;

    };

}
