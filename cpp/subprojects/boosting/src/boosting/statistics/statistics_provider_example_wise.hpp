#include "common/statistics/statistics_provider.hpp"
#include "boosting/statistics/statistics_example_wise.hpp"
#include "boosting/statistics/statistics_label_wise.hpp"


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

    /**
     * Provides access to an object of type `IExampleWiseStatistics` that can be converted into an object of type
     * `ILabelWiseStatistics`.
     */
    class ConvertibleExampleWiseStatisticsProvider : public IStatisticsProvider {

        private:

            const ILabelWiseRuleEvaluationFactory& regularRuleEvaluationFactory_;

            const ILabelWiseRuleEvaluationFactory& pruningRuleEvaluationFactory_;

            std::unique_ptr<IExampleWiseStatistics> exampleWiseStatisticsPtr_;

            std::unique_ptr<ILabelWiseStatistics> labelWiseStatisticsPtr_;

        public:

            /**
             * @param regularRuleEvaluationFactory  A reference to an object of type `ILabelWiseRuleEvaluationFactory`
             *                                      to switch to when invoking the function
             *                                      `switchToRegularRuleEvaluation`
             * @param pruningRuleEvaluationFactory  A reference to an object of type `ILabelWiseRuleEvaluationFactory`
             *                                      to switch to when invoking the function
             *                                      `switchToPruningRuleEvaluation`
             * @param statisticsPtr                 An unique pointer to an object of type `IExampleWiseStatistics` to
             *                                      provide access to
             */
            ConvertibleExampleWiseStatisticsProvider(
                    const ILabelWiseRuleEvaluationFactory& regularRuleEvaluationFactory,
                    const ILabelWiseRuleEvaluationFactory& pruningRuleEvaluationFactory,
                    std::unique_ptr<IExampleWiseStatistics> statisticsPtr)
                : regularRuleEvaluationFactory_(regularRuleEvaluationFactory),
                  pruningRuleEvaluationFactory_(pruningRuleEvaluationFactory),
                  exampleWiseStatisticsPtr_(std::move(statisticsPtr)) {

            }

            IStatistics& get() const override {
                IExampleWiseStatistics* exampleWiseStatistics = exampleWiseStatisticsPtr_.get();

                if (exampleWiseStatistics != nullptr) {
                    return *exampleWiseStatistics;
                } else {
                    return *labelWiseStatisticsPtr_;
                }
            }

            void switchToRegularRuleEvaluation() override {
                IExampleWiseStatistics* exampleWiseStatistics = exampleWiseStatisticsPtr_.get();

                if (exampleWiseStatistics != nullptr) {
                    labelWiseStatisticsPtr_ = exampleWiseStatistics->toLabelWiseStatistics(
                        regularRuleEvaluationFactory_);
                    exampleWiseStatisticsPtr_.reset();
                } else {
                    labelWiseStatisticsPtr_->setRuleEvaluationFactory(regularRuleEvaluationFactory_);
                }
            }

            void switchToPruningRuleEvaluation() override {
                IExampleWiseStatistics* exampleWiseStatistics = exampleWiseStatisticsPtr_.get();

                if (exampleWiseStatistics != nullptr) {
                    labelWiseStatisticsPtr_ = exampleWiseStatistics->toLabelWiseStatistics(
                        pruningRuleEvaluationFactory_);
                    exampleWiseStatisticsPtr_.reset();
                } else {
                    labelWiseStatisticsPtr_->setRuleEvaluationFactory(pruningRuleEvaluationFactory_);
                }
            }

    };

}
