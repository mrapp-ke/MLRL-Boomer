#include "common/statistics/statistics_provider.hpp"
#include "boosting/statistics/statistics_example_wise.hpp"


namespace boosting {

    /**
     * Provides access to an object of type `IExampleWiseStatistics`.
     */
    class ExampleWiseStatisticsProvider : public IStatisticsProvider {

        private:

            const IExampleWiseRuleEvaluationFactory& ruleEvaluationFactory_;

            std::unique_ptr<IExampleWiseStatistics> statisticsPtr_;

        public:

            /**
             * @param ruleEvaluationFactory A reference to an object of type `IExampleWiseRuleEvaluationFactory` to
             *                              switch to when invoking the function `switchRuleEvaluation`
             * @param statisticsPtr         An unique pointer to an object of type `IExampleWiseStatistics` to provide
             *                              access to
             */
            ExampleWiseStatisticsProvider(const IExampleWiseRuleEvaluationFactory& ruleEvaluationFactory,
                                          std::unique_ptr<IExampleWiseStatistics> statisticsPtr)
                : ruleEvaluationFactory_(ruleEvaluationFactory), statisticsPtr_(std::move(statisticsPtr)) {

            }

            IStatistics& get() const override {
                return *statisticsPtr_;
            }

            void switchRuleEvaluation() override {
                statisticsPtr_->setRuleEvaluationFactory(ruleEvaluationFactory_);
            }

    };

}
