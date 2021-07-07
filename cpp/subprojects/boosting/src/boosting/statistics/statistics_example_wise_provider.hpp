#include "common/statistics/statistics_provider.hpp"
#include "boosting/statistics/statistics_example_wise.hpp"


namespace boosting {

    /**
     * Provides access to an object of type `IExampleWiseStatistics`.
     */
    class ExampleWiseStatisticsProvider : public IStatisticsProvider {

        private:

            const IExampleWiseRuleEvaluationFactory& regularRuleEvaluationFactory_;

            const IExampleWiseRuleEvaluationFactory& pruningRuleEvaluationFactory_;

            std::unique_ptr<IExampleWiseStatistics> statisticsPtr_;

        public:

            /**
             * @param regularRuleEvaluationFactory  A reference to an object of type `IExampleWiseRuleEvaluationFactory`
             *                                      to switch to when invoking the function
             *                                      `switchToRegularRuleEvaluation`
             * @param pruningRuleEvaluationFactory  A reference to an object of type `IExampleWiseRuleEvaluationFactory`
             *                                      to switch to when invoking the function
             *                                      `switchToPruningRuleEvaluation`
             * @param statisticsPtr                 An unique pointer to an object of type `IExampleWiseStatistics` to
             *                                      provide access to
             */
            ExampleWiseStatisticsProvider(const IExampleWiseRuleEvaluationFactory& regularRuleEvaluationFactory,
                                          const IExampleWiseRuleEvaluationFactory& pruningRuleEvaluationFactory,
                                          std::unique_ptr<IExampleWiseStatistics> statisticsPtr)
                : regularRuleEvaluationFactory_(regularRuleEvaluationFactory),
                  pruningRuleEvaluationFactory_(pruningRuleEvaluationFactory),
                  statisticsPtr_(std::move(statisticsPtr)) {

            }

            IStatistics& get() const override {
                return *statisticsPtr_;
            }

            void switchToRegularRuleEvaluation() override {
                statisticsPtr_->setRuleEvaluationFactory(regularRuleEvaluationFactory_);
            }

            void switchToPruningRuleEvaluation() override {
                statisticsPtr_->setRuleEvaluationFactory(pruningRuleEvaluationFactory_);
            }

    };

}
