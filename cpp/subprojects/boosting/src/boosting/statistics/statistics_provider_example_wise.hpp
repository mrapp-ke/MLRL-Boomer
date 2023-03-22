/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/statistics/statistics_example_wise.hpp"
#include "boosting/statistics/statistics_label_wise.hpp"
#include "common/statistics/statistics_provider.hpp"

namespace boosting {

    /**
     * Provides access to an object of type `IExampleWiseStatistics`.
     *
     * @tparam LabelWiseRuleEvaluationFactory   The type of the classes that may be used for calculating the label-wise
     *                                          predictions of rules, as well as their overall quality
     * @tparam ExampleWiseRuleEvaluationFactory The type of the classes that may be used for calculating the
     *                                          example-wise predictions of rules, as well as their overall quality
     */
    template<typename ExampleWiseRuleEvaluationFactory, typename LabelWiseRuleEvaluationFactory>
    class ExampleWiseStatisticsProvider final : public IStatisticsProvider {
        private:

            typedef IExampleWiseStatistics<ExampleWiseRuleEvaluationFactory, LabelWiseRuleEvaluationFactory>
              ExampleWiseStatistics;

            const ExampleWiseRuleEvaluationFactory& regularRuleEvaluationFactory_;

            const ExampleWiseRuleEvaluationFactory& pruningRuleEvaluationFactory_;

            const std::unique_ptr<ExampleWiseStatistics> statisticsPtr_;

        public:

            /**
             * @param regularRuleEvaluationFactory  A reference to an object of template type
             *                                      `ExampleWiseRuleEvaluationFactory` to switch to when invoking the
             *                                      function `switchToRegularRuleEvaluation`
             * @param pruningRuleEvaluationFactory  A reference to an object of template type
             *                                      `ExampleWiseRuleEvaluationFactory` to switch to when invoking the
             *                                      function `switchToPruningRuleEvaluation`
             * @param statisticsPtr                 An unique pointer to an object of type `IExampleWiseStatistics` to
             *                                      provide access to
             */
            ExampleWiseStatisticsProvider(const ExampleWiseRuleEvaluationFactory& regularRuleEvaluationFactory,
                                          const ExampleWiseRuleEvaluationFactory& pruningRuleEvaluationFactory,
                                          std::unique_ptr<ExampleWiseStatistics> statisticsPtr)
                : regularRuleEvaluationFactory_(regularRuleEvaluationFactory),
                  pruningRuleEvaluationFactory_(pruningRuleEvaluationFactory),
                  statisticsPtr_(std::move(statisticsPtr)) {}

            /**
             * @see `IStatisticsProvider::get`
             */
            IStatistics& get() const override {
                return *statisticsPtr_;
            }

            /**
             * @see `IStatisticsProvider::switchToRegularRuleEvaluation`
             */
            void switchToRegularRuleEvaluation() override {
                statisticsPtr_->setRuleEvaluationFactory(regularRuleEvaluationFactory_);
            }

            /**
             * @see `IStatisticsProvider::switchToPruningRuleEvaluation`
             */
            void switchToPruningRuleEvaluation() override {
                statisticsPtr_->setRuleEvaluationFactory(pruningRuleEvaluationFactory_);
            }
    };

    /**
     * Provides access to an object of type `IExampleWiseStatistics` that can be converted into an object of type
     * `ILabelWiseStatistics`.
     *
     * @tparam LabelWiseRuleEvaluationFactory   The type of the classes that may be used for calculating the label-wise
     *                                          predictions of rules, as well as their overall quality
     * @tparam ExampleWiseRuleEvaluationFactory The type of the classes that may be used for calculating the
     *                                          example-wise predictions of rules, as well as their overall quality
     */
    template<typename ExampleWiseRuleEvaluationFactory, typename LabelWiseRuleEvaluationFactory>
    class ConvertibleExampleWiseStatisticsProvider final : public IStatisticsProvider {
        private:

            typedef IExampleWiseStatistics<ExampleWiseRuleEvaluationFactory, LabelWiseRuleEvaluationFactory>
              ExampleWiseStatistics;

            const LabelWiseRuleEvaluationFactory& regularRuleEvaluationFactory_;

            const LabelWiseRuleEvaluationFactory& pruningRuleEvaluationFactory_;

            std::unique_ptr<ExampleWiseStatistics> exampleWiseStatisticsPtr_;

            std::unique_ptr<ILabelWiseStatistics<LabelWiseRuleEvaluationFactory>> labelWiseStatisticsPtr_;

            const uint32 numThreads_;

        public:

            /**
             * @param regularRuleEvaluationFactory  A reference to an object of template type
             *                                      `LabelWiseRuleEvaluationFactory` to switch to when invoking the
             *                                      function `switchToRegularRuleEvaluation`
             * @param pruningRuleEvaluationFactory  A reference to an object of template type
             *                                      `LabelWiseRuleEvaluationFactory` to switch to when invoking the
             *                                      function `switchToPruningRuleEvaluation`
             * @param statisticsPtr                 An unique pointer to an object of type `IExampleWiseStatistics` to
             *                                      provide access to
             * @param numThreads                    The number of threads that should be used to convert the statistics
             *                                      for individual examples in parallel
             */
            ConvertibleExampleWiseStatisticsProvider(const LabelWiseRuleEvaluationFactory& regularRuleEvaluationFactory,
                                                     const LabelWiseRuleEvaluationFactory& pruningRuleEvaluationFactory,
                                                     std::unique_ptr<ExampleWiseStatistics> statisticsPtr,
                                                     uint32 numThreads)
                : regularRuleEvaluationFactory_(regularRuleEvaluationFactory),
                  pruningRuleEvaluationFactory_(pruningRuleEvaluationFactory),
                  exampleWiseStatisticsPtr_(std::move(statisticsPtr)), numThreads_(numThreads) {}

            /**
             * @see `IStatisticsProvider::get`
             */
            IStatistics& get() const override {
                ExampleWiseStatistics* exampleWiseStatistics = exampleWiseStatisticsPtr_.get();

                if (exampleWiseStatistics) {
                    return *exampleWiseStatistics;
                } else {
                    return *labelWiseStatisticsPtr_;
                }
            }

            /**
             * @see `IStatisticsProvider::switchToRegularRuleEvaluation`
             */
            void switchToRegularRuleEvaluation() override {
                ExampleWiseStatistics* exampleWiseStatistics = exampleWiseStatisticsPtr_.get();

                if (exampleWiseStatistics) {
                    labelWiseStatisticsPtr_ =
                      exampleWiseStatistics->toLabelWiseStatistics(regularRuleEvaluationFactory_, numThreads_);
                    exampleWiseStatisticsPtr_.reset();
                } else {
                    labelWiseStatisticsPtr_->setRuleEvaluationFactory(regularRuleEvaluationFactory_);
                }
            }

            /**
             * @see `IStatisticsProvider::switchToPruningRuleEvaluation`
             */
            void switchToPruningRuleEvaluation() override {
                ExampleWiseStatistics* exampleWiseStatistics = exampleWiseStatisticsPtr_.get();

                if (exampleWiseStatistics) {
                    labelWiseStatisticsPtr_ =
                      exampleWiseStatistics->toLabelWiseStatistics(pruningRuleEvaluationFactory_, numThreads_);
                    exampleWiseStatisticsPtr_.reset();
                } else {
                    labelWiseStatisticsPtr_->setRuleEvaluationFactory(pruningRuleEvaluationFactory_);
                }
            }
    };

}
