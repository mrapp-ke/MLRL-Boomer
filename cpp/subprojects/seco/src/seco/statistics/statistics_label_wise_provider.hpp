#include "common/statistics/statistics_provider_factory.hpp"
#include "seco/statistics/statistics_label_wise.hpp"


namespace seco {

    /**
     * Provides access to an object of type `ILabelWiseStatistics`.
     */
    class LabelWiseStatisticsProvider : public IStatisticsProvider {

        private:

            const ILabelWiseRuleEvaluationFactory& regularRuleEvaluationFactory_;

            std::unique_ptr<ILabelWiseStatistics> statisticsPtr_;

        public:

            /**
             * @param regularRuleEvaluationFactory  A reference to an object of type `ILabelWiseRuleEvaluationFactory`
             *                                      to switch to when invoking the function
             *                                      `switchToRegularRuleEvaluation`
             * @param statisticsPtr                 An unique pointer to an object of type `ILabelWiseStatistics` to
             *                                      provide access to
             */
            LabelWiseStatisticsProvider(const ILabelWiseRuleEvaluationFactory& regularRuleEvaluationFactory,
                                        std::unique_ptr<ILabelWiseStatistics> statisticsPtr)
                : regularRuleEvaluationFactory_(regularRuleEvaluationFactory),
                  statisticsPtr_(std::move(statisticsPtr)) {

            }

            IStatistics& get() const override {
                return *statisticsPtr_;
            }

            void switchToRegularRuleEvaluation() override {
                statisticsPtr_->setRuleEvaluationFactory(regularRuleEvaluationFactory_);
            }

    };

}
