/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/statistics/statistics.hpp"
#include "mlrl/common/statistics/statistics_weighted.hpp"
#include "statistics.hpp"
#include "statistics_subset_resettable.hpp"

#include <memory>
#include <utility>

namespace boosting {

    /**
     * Provides access to weighted gradients and Hessians that are calculated according to a loss function and allows to
     * update the gradients and Hessians after a new rule has been learned.
     *
     * @tparam State                    The type of the state of the training process
     * @tparam StatisticVector          The type of the vectors that are used to store statistics
     * @tparam WeightVector             The type of the vector that provides access to the weights of individual
     *                                  statistics
     * @tparam RuleEvaluationFactory    The type of the factory that allows to create instances of the class that is
     *                                  used for calculating the predictions of rules, as well as corresponding quality
     *                                  scores
     */
    template<typename State, typename StatisticVector, typename WeightVector, typename RuleEvaluationFactory>
    class WeightedStatistics final : public AbstractWeightedStatistics<State, StatisticVector, WeightVector> {
        private:

            template<typename IndexVector>
            using StatisticsSubset = ResettableBoostingStatisticsSubset<State, StatisticVector, WeightVector,
                                                                        IndexVector, RuleEvaluationFactory>;

            const RuleEvaluationFactory& ruleEvaluationFactory_;

        public:

            /**
             * @param state                 A reference to an object of template type `State` that represents the state
             *                              of the training process
             * @param weights               A reference to an object of template type `WeightVector` that provides
             *                              access to the weights of individual statistics
             * @param ruleEvaluationFactory A reference to an object of template type `RuleEvaluationFactory` that
             *                              allows to create instances of the class that should be used for calculating
             *                              the predictions of rules, as well as their overall quality
             */
            WeightedStatistics(State& state, const WeightVector& weights,
                               const RuleEvaluationFactory& ruleEvaluationFactory)
                : AbstractWeightedStatistics<State, StatisticVector, WeightVector>(state, weights),
                  ruleEvaluationFactory_(ruleEvaluationFactory) {}

            /**
             * @param other A reference to an object of type `WeightedStatistics` to be copied
             */
            WeightedStatistics(const WeightedStatistics& other)
                : AbstractWeightedStatistics<State, StatisticVector, WeightVector>(other),
                  ruleEvaluationFactory_(other.ruleEvaluationFactory_) {}

            /**
             * @see `IWeightedStatistics::copy`
             */
            std::unique_ptr<IWeightedStatistics> copy() const override {
                return std::make_unique<
                  WeightedStatistics<State, StatisticVector, WeightVector, RuleEvaluationFactory>>(*this);
            }

            /**
             * @see `IWeightedStatistics::createSubset`
             */
            std::unique_ptr<IResettableStatisticsSubset> createSubset(
              const BinaryDokVector& excludedStatisticIndices,
              const CompleteIndexVector& outputIndices) const override {
                return std::make_unique<StatisticsSubset<CompleteIndexVector>>(
                  this->state_, this->weights_, outputIndices, ruleEvaluationFactory_, this->totalSumVector_,
                  excludedStatisticIndices);
            }

            /**
             * @see `IWeightedStatistics::createSubset`
             */
            std::unique_ptr<IResettableStatisticsSubset> createSubset(
              const BinaryDokVector& excludedStatisticIndices, const PartialIndexVector& outputIndices) const override {
                return std::make_unique<StatisticsSubset<PartialIndexVector>>(
                  this->state_, this->weights_, outputIndices, ruleEvaluationFactory_, this->totalSumVector_,
                  excludedStatisticIndices);
            }
    };
}
