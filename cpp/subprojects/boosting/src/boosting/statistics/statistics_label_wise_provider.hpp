#include "common/statistics/statistics_provider.hpp"
#include "boosting/statistics/statistics_label_wise.hpp"


namespace boosting {

    /**
     * Provides access to an object of type `ILabelWiseStatistics`.
     */
    class LabelWiseStatisticsProvider : public IStatisticsProvider {

        private:

            const ILabelWiseRuleEvaluationFactory& ruleEvaluationFactory_;

            std::unique_ptr<ILabelWiseStatistics> statisticsPtr_;

        public:

            /**
             * @param ruleEvaluationFactory A reference to an object of type `ILabelWiseRuleEvaluationFactory` to switch
             *                              to when invoking the function `switchRuleEvaluation`
             * @param statisticsPtr         An unique pointer to an object of type `ILabelWiseStatistics` to provide
             *                              access to
             */
            LabelWiseStatisticsProvider(const ILabelWiseRuleEvaluationFactory& ruleEvaluationFactory,
                                        std::unique_ptr<ILabelWiseStatistics> statisticsPtr)
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
