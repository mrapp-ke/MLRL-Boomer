/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/statistics/statistics_provider.hpp"
#include "mlrl/seco/statistics/statistics_decomposable.hpp"

#include <memory>
#include <utility>

#include <memory>
#include <utility>

namespace seco {

    /**
     * Provides access to an object of type `IDecomposableStatistics`.
     *
     * @tparam RuleEvaluationFactory The type of the classes that may be used for calculating the predictions of rules,
     *                               as well as their overall quality
     */
    template<typename RuleEvaluationFactory>
    class DecomposableStatisticsProvider final : public IStatisticsProvider {
        private:

            const RuleEvaluationFactory& regularRuleEvaluationFactory_;

            const RuleEvaluationFactory& pruningRuleEvaluationFactory_;

            const std::unique_ptr<IDecomposableStatistics<RuleEvaluationFactory>> statisticsPtr_;

        public:

            /**
             * @param regularRuleEvaluationFactory  A reference to an object of template type `RuleEvaluationFactory` to
             *                                      switch to when invoking the function `switchToRegularRuleEvaluation`
             * @param pruningRuleEvaluationFactory  A reference to an object of template type `RuleEvaluationFactory` to
             *                                      switch to when invoking the function `switchToPruningRuleEvaluation`
             * @param statisticsPtr                 An unique pointer to an object of type `IDecomposableStatistics` to
             *                                      provide access to
             */
            DecomposableStatisticsProvider(
              const RuleEvaluationFactory& regularRuleEvaluationFactory,
              const RuleEvaluationFactory& pruningRuleEvaluationFactory,
              std::unique_ptr<IDecomposableStatistics<RuleEvaluationFactory>> statisticsPtr)
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

}
