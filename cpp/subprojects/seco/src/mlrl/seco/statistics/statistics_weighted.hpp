/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/statistics/statistics_weighted.hpp"
#include "mlrl/seco/statistics/statistics.hpp"
#include "statistics.hpp"
#include "statistics_subset.hpp"

#include <memory>
#include <utility>

namespace seco {

    /**
     * An abstract base class for all statistics that provide access to the elements of weighted confusion matrices.
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
            using StatisticsSubset = ResettableCoverageStatisticsSubset<State, StatisticVector, WeightVector,
                                                                        IndexVector, RuleEvaluationFactory>;

            const RuleEvaluationFactory& ruleEvaluationFactory_;

            StatisticVector subsetSumVector_;

        public:

            /**
             * @param state                 A reference to an object of template type `State` that represents the state
             *                              of the training process
             * @param ruleEvaluationFactory A reference to an object of template type `RuleEvaluationFactory` that
             *                              allows to create instances of the class that should be used for calculating
             *                              the predictions of rules, as well as their overall quality
             * @param weights               A reference to an object of template type `WeightVector` that provides
             *                              access to the weights of individual statistics
             */
            WeightedStatistics(State& state, const RuleEvaluationFactory& ruleEvaluationFactory,
                               const WeightVector& weights)
                : AbstractWeightedStatistics<State, StatisticVector, WeightVector>(state, weights),
                  ruleEvaluationFactory_(ruleEvaluationFactory),
                  subsetSumVector_(state.statisticMatrixPtr->getNumCols(), true) {
                this->initializeSumVector(weights, state.statisticMatrixPtr->getView(), subsetSumVector_);
            }

            /**
             * @param other A reference to an object of type `WeightedStatistics` to be copied
             */
            WeightedStatistics(const WeightedStatistics& other)
                : AbstractWeightedStatistics<State, StatisticVector, WeightVector>(other),
                  ruleEvaluationFactory_(other.ruleEvaluationFactory_), subsetSumVector_(other.subsetSumVector_) {}

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
                  this->state_, this->weights_, outputIndices, ruleEvaluationFactory_, subsetSumVector_,
                  this->totalSumVector_, excludedStatisticIndices);
            }

            /**
             * @see `IWeightedStatistics::createSubset`
             */
            std::unique_ptr<IResettableStatisticsSubset> createSubset(
              const BinaryDokVector& excludedStatisticIndices, const PartialIndexVector& outputIndices) const override {
                return std::make_unique<StatisticsSubset<PartialIndexVector>>(
                  this->state_, this->weights_, outputIndices, ruleEvaluationFactory_, subsetSumVector_,
                  this->totalSumVector_, excludedStatisticIndices);
            }
    };
}
