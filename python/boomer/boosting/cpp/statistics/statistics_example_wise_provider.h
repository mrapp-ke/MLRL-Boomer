/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../../common/cpp/statistics/statistics_provider.h"
#include "../losses/loss_example_wise.h"
#include "statistics_example_wise.h"


namespace boosting {

    /**
     * Allows to create instances of the class `ExampleWiseStatisticsProvider`.
     */
    class ExampleWiseStatisticsProviderFactory: public IStatisticsProviderFactory {

        private:

            std::shared_ptr<IExampleWiseLoss> lossFunctionPtr_;

            std::shared_ptr<IExampleWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr_;

            std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr_;

        public:

            /**
             * @param lossFunctionPtr                   A shared pointer to an object of type `IExampleWiseLoss` that
             *                                          should be used for calculating gradients and Hessians
             * @param defaultRuleEvaluationFactoryPtr   A shared pointer to an object of type
             *                                          `IExampleWiseRuleEvaluationFactory` that should be used for
             *                                          calculating the predictions, as well as corresponding quality
             *                                          scores, of the default rule
             * @param ruleEvaluationFactoryPtr          A shared pointer to an object of type
             *                                          `IExampleWiseRuleEvaluationFactory` that should be used for
             *                                          calculating the predictions, as well as corresponding quality
             *                                          scores, of all remaining rules
             */
            ExampleWiseStatisticsProviderFactory(
                std::shared_ptr<IExampleWiseLoss> lossFunctionPtr,
                std::shared_ptr<IExampleWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
                std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr);

            std::unique_ptr<IStatisticsProvider> create(
                std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr) const override;

    };

}
