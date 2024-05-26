/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/statistics/statistics_decomposable.hpp"
#include "mlrl/boosting/statistics/statistics_non_decomposable.hpp"
#include "mlrl/common/statistics/statistics_provider.hpp"

namespace boosting {

    /**
     * Provides access to an object of type `INonDecomposableStatistics`.
     *
     * @tparam LabelWiseRuleEvaluationFactory   The type of the classes that may be used for calculating the label-wise
     *                                          predictions of rules, as well as their overall quality
     * @tparam ExampleWiseRuleEvaluationFactory The type of the classes that may be used for calculating the
     *                                          example-wise predictions of rules, as well as their overall quality
     */
    template<typename ExampleWiseRuleEvaluationFactory, typename LabelWiseRuleEvaluationFactory>
    class NonDecomposableStatisticsProvider final : public IStatisticsProvider {
        private:

            typedef INonDecomposableStatistics<ExampleWiseRuleEvaluationFactory, LabelWiseRuleEvaluationFactory>
              NonDecomposableStatistics;

            const ExampleWiseRuleEvaluationFactory& regularRuleEvaluationFactory_;

            const ExampleWiseRuleEvaluationFactory& pruningRuleEvaluationFactory_;

            const std::unique_ptr<NonDecomposableStatistics> statisticsPtr_;

        public:

            /**
             * @param regularRuleEvaluationFactory  A reference to an object of template type
             *                                      `ExampleWiseRuleEvaluationFactory` to switch to when invoking the
             *                                      function `switchToRegularRuleEvaluation`
             * @param pruningRuleEvaluationFactory  A reference to an object of template type
             *                                      `ExampleWiseRuleEvaluationFactory` to switch to when invoking the
             *                                      function `switchToPruningRuleEvaluation`
             * @param statisticsPtr                 An unique pointer to an object of type `INonDecomposableStatistics`
             *                                      to provide access to
             */
            NonDecomposableStatisticsProvider(const ExampleWiseRuleEvaluationFactory& regularRuleEvaluationFactory,
                                              const ExampleWiseRuleEvaluationFactory& pruningRuleEvaluationFactory,
                                              std::unique_ptr<NonDecomposableStatistics> statisticsPtr)
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
     * Provides access to an object of type `INonDecomposableStatistics` that can be converted into an object of type
     * `IDecomposableStatistics`.
     *
     * @tparam LabelWiseRuleEvaluationFactory   The type of the classes that may be used for calculating the label-wise
     *                                          predictions of rules, as well as their overall quality
     * @tparam ExampleWiseRuleEvaluationFactory The type of the classes that may be used for calculating the
     *                                          example-wise predictions of rules, as well as their overall quality
     */
    template<typename ExampleWiseRuleEvaluationFactory, typename LabelWiseRuleEvaluationFactory>
    class ConvertibleNonDecomposableStatisticsProvider final : public IStatisticsProvider {
        private:

            typedef INonDecomposableStatistics<ExampleWiseRuleEvaluationFactory, LabelWiseRuleEvaluationFactory>
              NonDecomposableStatistics;

            const LabelWiseRuleEvaluationFactory& regularRuleEvaluationFactory_;

            const LabelWiseRuleEvaluationFactory& pruningRuleEvaluationFactory_;

            std::unique_ptr<NonDecomposableStatistics> nonDecomposableStatisticsPtr_;

            std::unique_ptr<IDecomposableStatistics<LabelWiseRuleEvaluationFactory>> decomposableStatisticsPtr_;

            const uint32 numThreads_;

        public:

            /**
             * @param regularRuleEvaluationFactory  A reference to an object of template type
             *                                      `LabelWiseRuleEvaluationFactory` to switch to when invoking the
             *                                      function `switchToRegularRuleEvaluation`
             * @param pruningRuleEvaluationFactory  A reference to an object of template type
             *                                      `LabelWiseRuleEvaluationFactory` to switch to when invoking the
             *                                      function `switchToPruningRuleEvaluation`
             * @param statisticsPtr                 An unique pointer to an object of type `INonDecomposableStatistics`
             *                                      to provide access to
             * @param numThreads                    The number of threads that should be used to convert the statistics
             *                                      for individual examples in parallel
             */
            ConvertibleNonDecomposableStatisticsProvider(
              const LabelWiseRuleEvaluationFactory& regularRuleEvaluationFactory,
              const LabelWiseRuleEvaluationFactory& pruningRuleEvaluationFactory,
              std::unique_ptr<NonDecomposableStatistics> statisticsPtr, uint32 numThreads)
                : regularRuleEvaluationFactory_(regularRuleEvaluationFactory),
                  pruningRuleEvaluationFactory_(pruningRuleEvaluationFactory),
                  nonDecomposableStatisticsPtr_(std::move(statisticsPtr)), numThreads_(numThreads) {}

            /**
             * @see `IStatisticsProvider::get`
             */
            IStatistics& get() const override {
                NonDecomposableStatistics* nonDecomposableStatistics = nonDecomposableStatisticsPtr_.get();

                if (nonDecomposableStatistics) {
                    return *nonDecomposableStatistics;
                } else {
                    return *decomposableStatisticsPtr_;
                }
            }

            /**
             * @see `IStatisticsProvider::switchToRegularRuleEvaluation`
             */
            void switchToRegularRuleEvaluation() override {
                NonDecomposableStatistics* nonDecomposableStatistics = nonDecomposableStatisticsPtr_.get();

                if (nonDecomposableStatistics) {
                    decomposableStatisticsPtr_ =
                      nonDecomposableStatistics->toDecomposableStatistics(regularRuleEvaluationFactory_, numThreads_);
                    nonDecomposableStatisticsPtr_.reset();
                } else {
                    decomposableStatisticsPtr_->setRuleEvaluationFactory(regularRuleEvaluationFactory_);
                }
            }

            /**
             * @see `IStatisticsProvider::switchToPruningRuleEvaluation`
             */
            void switchToPruningRuleEvaluation() override {
                NonDecomposableStatistics* nonDecomposableStatistics = nonDecomposableStatisticsPtr_.get();

                if (nonDecomposableStatistics) {
                    decomposableStatisticsPtr_ =
                      nonDecomposableStatistics->toDecomposableStatistics(pruningRuleEvaluationFactory_, numThreads_);
                    nonDecomposableStatisticsPtr_.reset();
                } else {
                    decomposableStatisticsPtr_->setRuleEvaluationFactory(pruningRuleEvaluationFactory_);
                }
            }
    };

}
