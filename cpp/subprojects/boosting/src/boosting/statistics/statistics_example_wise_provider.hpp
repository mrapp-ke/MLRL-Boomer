#include "common/statistics/statistics_provider.hpp"
#include "boosting/statistics/statistics_example_wise.hpp"


namespace boosting {

    /**
     * Provides access to an object of type `IExampleWiseStatistics`.
     */
    class ExampleWiseStatisticsProvider : public IStatisticsProvider {

        private:

            std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr_;

            std::unique_ptr<IExampleWiseStatistics> statisticsPtr_;

        public:

            /**
             * @param ruleEvaluationFactoryPtr  A shared pointer to an object of type
             *                                  `IExampleWiseRuleEvaluationFactory` to switch to when invoking the
             *                                  function `switchRuleEvaluation`
             * @param statisticsPtr             An unique pointer to an object of type `IExampleWiseStatistics` to
             *                                  provide access to
             */
            ExampleWiseStatisticsProvider(std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
                                          std::unique_ptr<IExampleWiseStatistics> statisticsPtr)
                : ruleEvaluationFactoryPtr_(ruleEvaluationFactoryPtr), statisticsPtr_(std::move(statisticsPtr)) {

            }

            IStatistics& get() const override {
                return *statisticsPtr_;
            }

            void switchRuleEvaluation() override {
                statisticsPtr_->setRuleEvaluationFactory(ruleEvaluationFactoryPtr_);
            }

    };

}
