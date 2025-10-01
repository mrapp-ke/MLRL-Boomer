/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/statistics/statistics.hpp"
#include "statistics.hpp"
#include "statistics_subset.hpp"

#include <memory>
#include <utility>

namespace boosting {

    template<typename StatisticView, typename StatisticVector>
    static inline void addStatisticInternally(const EqualWeightVector& weights, const StatisticView& statisticView,
                                              StatisticVector& statisticVector, uint32 statisticIndex) {
        statisticVector.add(statisticView, statisticIndex);
    }

    template<typename WeightVector, typename StatisticView, typename StatisticVector>
    static inline void addStatisticInternally(const WeightVector& weights, const StatisticView& statisticView,
                                              StatisticVector& statisticVector, uint32 statisticIndex) {
        typename WeightVector::weight_type weight = weights[statisticIndex];
        statisticVector.add(statisticView, statisticIndex, weight);
    }

    template<typename StatisticView, typename StatisticVector>
    static inline void removeStatisticInternally(const EqualWeightVector& weights, const StatisticView& statisticView,
                                                 StatisticVector& statisticVector, uint32 statisticIndex) {
        statisticVector.remove(statisticView, statisticIndex);
    }

    template<typename WeightVector, typename StatisticView, typename StatisticVector>
    static inline void removeStatisticInternally(const WeightVector& weights, const StatisticView& statisticView,
                                                 StatisticVector& statisticVector, uint32 statisticIndex) {
        typename WeightVector::weight_type weight = weights[statisticIndex];
        statisticVector.remove(statisticView, statisticIndex, weight);
    }

    /**
     * Provides access to weighted gradients and Hessians that are calculated according to a loss function and allows to
     * update the gradients and Hessians after a new rule has been learned.
     *
     * @tparam State                    The type of the state of the training process
     * @tparam StatisticVector          The type of the vectors that are used to store statistics
     * @tparam RuleEvaluationFactory    The type of the factory that allows to create instances of the class that is
     *                                  used for calculating the predictions of rules, as well as corresponding quality
     *                                  scores
     * @tparam WeightVector             The type of the vector that provides access to the weights of individual
     *                                  statistics
     */
    template<typename State, typename StatisticVector, typename RuleEvaluationFactory, typename WeightVector>
    class WeightedStatistics final : public AbstractStatisticsSpace<State>,
                                     virtual public IWeightedStatistics {
        private:

            template<typename IndexVector>
            using StatisticsSubset = ResettableBoostingStatisticsSubset<State, StatisticVector, WeightVector,
                                                                        IndexVector, RuleEvaluationFactory>;

            const WeightVector& weights_;

            const RuleEvaluationFactory& ruleEvaluationFactory_;

            const std::unique_ptr<StatisticVector> totalSumVectorPtr_;

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
                : AbstractStatisticsSpace<State>(state), weights_(weights),
                  ruleEvaluationFactory_(ruleEvaluationFactory),
                  totalSumVectorPtr_(std::make_unique<StatisticVector>(state.statisticMatrixPtr->getNumCols(), true)) {
                uint32 numStatistics = weights.getNumElements();

                for (uint32 i = 0; i < numStatistics; i++) {
                    addStatisticInternally(weights, state.statisticMatrixPtr->getView(), *totalSumVectorPtr_, i);
                }
            }

            /**
             * @param statistics A reference to an object of type `WeightedStatistics` to be copied
             */
            WeightedStatistics(const WeightedStatistics& statistics)
                : AbstractStatisticsSpace<State>(statistics.state_), weights_(statistics.weights_),
                  ruleEvaluationFactory_(statistics.ruleEvaluationFactory_),
                  totalSumVectorPtr_(std::make_unique<StatisticVector>(*statistics.totalSumVectorPtr_)) {}

            /**
             * @see `IWeightedStatistics::copy`
             */
            std::unique_ptr<IWeightedStatistics> copy() const override {
                return std::make_unique<
                  WeightedStatistics<State, StatisticVector, RuleEvaluationFactory, WeightVector>>(*this);
            }

            /**
             * @see `IWeightedStatistics::resetCoveredStatistics`
             */
            void resetCoveredStatistics() override {
                totalSumVectorPtr_->clear();
            }

            /**
             * @see `IWeightedStatistics::addCoveredStatistic`
             */
            void addCoveredStatistic(uint32 statisticIndex) override {
                addStatisticInternally(this->weights_, this->state_.statisticMatrixPtr->getView(), *totalSumVectorPtr_,
                                       statisticIndex);
            }

            /**
             * @see `IWeightedStatistics::removeCoveredStatistic`
             */
            void removeCoveredStatistic(uint32 statisticIndex) override {
                removeStatisticInternally(this->weights_, this->state_.statisticMatrixPtr->getView(),
                                          *totalSumVectorPtr_, statisticIndex);
            }

            /**
             * @see `IWeightedStatistics::createSubset`
             */
            std::unique_ptr<IResettableStatisticsSubset> createSubset(
              const BinaryDokVector& excludedStatisticIndices,
              const CompleteIndexVector& outputIndices) const override {
                return std::make_unique<StatisticsSubset<CompleteIndexVector>>(
                  this->state_, weights_, outputIndices, ruleEvaluationFactory_, *totalSumVectorPtr_,
                  excludedStatisticIndices);
            }

            /**
             * @see `IWeightedStatistics::createSubset`
             */
            std::unique_ptr<IResettableStatisticsSubset> createSubset(
              const BinaryDokVector& excludedStatisticIndices, const PartialIndexVector& outputIndices) const override {
                return std::make_unique<StatisticsSubset<PartialIndexVector>>(
                  this->state_, weights_, outputIndices, ruleEvaluationFactory_, *totalSumVectorPtr_,
                  excludedStatisticIndices);
            }
    };
}
