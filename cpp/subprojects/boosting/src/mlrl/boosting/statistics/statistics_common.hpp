/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/statistics/statistics.hpp"
#include "statistics.hpp"
#include "statistics_space.hpp"

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
    class WeightedStatistics final
        : virtual public IWeightedStatistics,
          public AbstractBoostingStatisticsSpace<State, StatisticVector, WeightVector, RuleEvaluationFactory> {
        private:

            /**
             * Provides access to a subset of the statistics that are stored by an instance of the class
             * `WeightedStatistics`.
             *
             * @tparam IndexVector The type of the vector that provides access to the indices of the outputs that are
             *                     included in the subset
             */
            template<typename IndexVector>
            class StatisticsSubset final
                : public AbstractBoostingStatisticsSpace<State, StatisticVector, WeightVector, RuleEvaluationFactory>::
                    template AbstractStatisticsSubset<IndexVector> {
                private:

                    std::unique_ptr<StatisticVector> totalCoverableSumVectorPtr_;

                public:

                    /**
                     * @param statistics                A reference to an object of type `WeightedStatistics` that
                     *                                  stores the statistics
                     * @param totalSumVector            A reference to an object of template type `StatisticVector` that
                     *                                  stores the total sums of gradients and Hessians
                     * @param excludedStatisticIndices  A reference to an object of type `BinaryDokVector` that provides
                     *                                  access to the indices of the statistics that should be excluded
                     *                                  from the subset
                     * @param outputIndices             A reference to an object of template type `IndexVector` that
                     *                                  provides access to the indices of the outputs that are included
                     *                                  in the subset
                     */
                    StatisticsSubset(const WeightedStatistics& statistics, const StatisticVector& totalSumVector,
                                     const BinaryDokVector& excludedStatisticIndices, const IndexVector& outputIndices)
                        : AbstractBoostingStatisticsSpace<State, StatisticVector, WeightVector, RuleEvaluationFactory>::
                            template AbstractStatisticsSubset<IndexVector>(statistics, totalSumVector, outputIndices) {
                        if (excludedStatisticIndices.getNumIndices() > 0) {
                            // Create a vector for storing the totals sums of gradients and Hessians, if necessary...
                            totalCoverableSumVectorPtr_ = std::make_unique<StatisticVector>(*this->totalSumVector_);
                            this->totalSumVector_ = totalCoverableSumVectorPtr_.get();

                            for (auto it = excludedStatisticIndices.indices_cbegin();
                                 it != excludedStatisticIndices.indices_cend(); it++) {
                                // Subtract the gradients and Hessians of the example at the given index (weighted by
                                // the given weight) from the total sums of gradients and Hessians...
                                uint32 statisticIndex = *it;
                                removeStatisticInternally(this->weights_, this->state_.statisticMatrixPtr->getView(),
                                                          *totalCoverableSumVectorPtr_, statisticIndex);
                            }
                        }
                    }
            };

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
                : AbstractBoostingStatisticsSpace<State, StatisticVector, WeightVector, RuleEvaluationFactory>(
                    state, ruleEvaluationFactory, weights),
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
                : AbstractBoostingStatisticsSpace<State, StatisticVector, WeightVector, RuleEvaluationFactory>(
                    statistics.state_, statistics.ruleEvaluationFactory_, statistics.weights_),
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
                return std::make_unique<StatisticsSubset<CompleteIndexVector>>(*this, *totalSumVectorPtr_,
                                                                               excludedStatisticIndices, outputIndices);
            }

            /**
             * @see `IWeightedStatistics::createSubset`
             */
            std::unique_ptr<IResettableStatisticsSubset> createSubset(
              const BinaryDokVector& excludedStatisticIndices, const PartialIndexVector& outputIndices) const override {
                return std::make_unique<StatisticsSubset<PartialIndexVector>>(*this, *totalSumVectorPtr_,
                                                                              excludedStatisticIndices, outputIndices);
            }
    };
}
