/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "boosting/rule_evaluation/rule_evaluation_example_wise.hpp"
#include "boosting/data/statistic_vector_dense_example_wise.hpp"


namespace boosting {

    /**
     * Allows to create instances of the class `ExampleWiseSingleLabelRuleEvaluationFactory`.
     */
    class ExampleWiseSingleLabelRuleEvaluationFactory final : public IExampleWiseRuleEvaluationFactory {

        private:

            float64 l2RegularizationWeight_;

        public:

            /**
             * @param l2RegularizationWeight The weight of the L2 regularization that is applied for calculating the
             *                               scores to be predicted by rules
             */
            ExampleWiseSingleLabelRuleEvaluationFactory(float64 l2RegularizationWeight);

            std::unique_ptr<IRuleEvaluation<DenseExampleWiseStatisticVector>> createDense(
                const CompleteIndexVector& indexVector) const override;

            std::unique_ptr<IRuleEvaluation<DenseExampleWiseStatisticVector>> createDense(
                const PartialIndexVector& indexVector) const override;

    };

}
