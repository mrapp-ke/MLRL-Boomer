/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/vector_sparse_array_binary.hpp"
#include "mlrl/seco/statistics/statistics.hpp"
#include "statistics_subset.hpp"

#include <memory>
#include <utility>

namespace seco {

    template<typename StatisticView, typename StatisticVector>
    static inline void initializeStatisticVector(const EqualWeightVector& weights, const StatisticView& statisticView,
                                                 StatisticVector& statisticVector) {
        uint32 numStatistics = weights.getNumElements();

        for (uint32 i = 0; i < numStatistics; i++) {
            statisticVector.add(statisticView, i);
        }
    }

    template<typename WeightVector, typename StatisticView, typename StatisticVector>
    static inline void initializeStatisticVector(const WeightVector& weights, const StatisticView& statisticView,
                                                 StatisticVector& statisticVector) {
        uint32 numStatistics = weights.getNumElements();

        for (uint32 i = 0; i < numStatistics; i++) {
            typename WeightVector::weight_type weight = weights[i];
            statisticVector.add(statisticView, i, weight);
        }
    }

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
     * An abstract base class for all statistics that provide access to the elements of weighted confusion matrices.
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

            /**
             * Provides access to a subset of the statistics that are stored by an instance of the class
             * `WeightedStatistics`.
             *
             * @tparam IndexVector The type of the vector that provides access to the indices of the outputs that are
             *                     included in the subset
             */
            template<typename IndexVector>
            class StatisticsSubset final : virtual public IResettableStatisticsSubset,
                                           public AbstractCoverageStatisticsSubset<State, StatisticVector, WeightVector,
                                                                                   IndexVector, RuleEvaluationFactory> {
                private:

                    const StatisticVector* subsetSumVector_;

                    StatisticVector tmpVector_;

                    std::unique_ptr<StatisticVector> accumulatedSumVectorPtr_;

                    std::unique_ptr<StatisticVector> totalCoverableSumVectorPtr_;

                public:

                    /**
                     * @param statistics                A reference to an object of type `WeightedStatistics` that
                     *                                  stores the statistics
                     * @param excludedStatisticIndices  A reference to an object of type `BinaryDokVector` that provides
                     *                                  access to the indices of the statistics that should be excluded
                     *                                  from the subset
                     * @param outputIndices             A reference to an object of template type `IndexVector` that
                     *                                  provides access to the indices of the outputs that are included
                     *                                  in the subset
                     */
                    StatisticsSubset(const WeightedStatistics& statistics,
                                     const BinaryDokVector& excludedStatisticIndices, const IndexVector& outputIndices)
                        : AbstractCoverageStatisticsSubset<State, StatisticVector, WeightVector, IndexVector,
                                                           RuleEvaluationFactory>(
                            statistics.state_, statistics.weights_, outputIndices, statistics.ruleEvaluationFactory_,
                            statistics.totalSumVector_),
                          subsetSumVector_(&statistics.subsetSumVector_), tmpVector_(outputIndices.getNumElements()) {
                        if (excludedStatisticIndices.getNumIndices() > 0) {
                            // Allocate a vector for storing the totals sums of confusion matrices, if necessary...
                            totalCoverableSumVectorPtr_ = std::make_unique<StatisticVector>(*subsetSumVector_);
                            subsetSumVector_ = totalCoverableSumVectorPtr_.get();

                            for (auto it = excludedStatisticIndices.indices_cbegin();
                                 it != excludedStatisticIndices.indices_cend(); it++) {
                                // For each output, subtract the confusion matrices of the example at the given index
                                // (weighted
                                // by the given weight) from the total sum of confusion matrices...
                                uint32 statisticIndex = *it;
                                removeStatisticInternally(this->weights_, this->state_.statisticMatrixPtr->getView(),
                                                          *totalCoverableSumVectorPtr_, statisticIndex);
                            }
                        }
                    }

                    /**
                     * @see `IResettableStatisticsSubset::resetSubset`
                     */
                    void resetSubset() override {
                        if (!accumulatedSumVectorPtr_) {
                            // Allocate a vector for storing the accumulated confusion matrices, if necessary...
                            accumulatedSumVectorPtr_ = std::make_unique<StatisticVector>(this->sumVector_);
                        } else {
                            // Add the confusion matrix for each output to the accumulated confusion matrix...
                            accumulatedSumVectorPtr_->add(this->sumVector_);
                        }

                        // Reset the confusion matrix for each output to zero...
                        this->sumVector_.clear();
                    }

                    /**
                     * @see `IResettableStatisticsSubset::calculateScoresAccumulated`
                     */
                    std::unique_ptr<IStatisticsUpdateCandidate> calculateScoresAccumulated() override {
                        const IScoreVector& scoreVector = this->ruleEvaluationPtr_->calculateScores(
                          this->state_.statisticMatrixPtr->majorityLabelVectorPtr->cbegin(),
                          this->state_.statisticMatrixPtr->majorityLabelVectorPtr->cend(), this->totalSumVector_,
                          *accumulatedSumVectorPtr_);
                        return this->state_.createUpdateCandidate(scoreVector);
                    }

                    /**
                     * @see `IResettableStatisticsSubset::calculateScoresUncovered`
                     */
                    std::unique_ptr<IStatisticsUpdateCandidate> calculateScoresUncovered() override {
                        tmpVector_.difference(subsetSumVector_->cbegin(), subsetSumVector_->cend(),
                                              this->outputIndices_, this->sumVector_.cbegin(), this->sumVector_.cend());
                        const IScoreVector& scoreVector = this->ruleEvaluationPtr_->calculateScores(
                          this->state_.statisticMatrixPtr->majorityLabelVectorPtr->cbegin(),
                          this->state_.statisticMatrixPtr->majorityLabelVectorPtr->cend(), this->totalSumVector_,
                          tmpVector_);
                        return this->state_.createUpdateCandidate(scoreVector);
                    }

                    /**
                     * @see `IResettableStatisticsSubset::calculateScoresUncoveredAccumulated`
                     */
                    std::unique_ptr<IStatisticsUpdateCandidate> calculateScoresUncoveredAccumulated() override {
                        tmpVector_.difference(subsetSumVector_->cbegin(), subsetSumVector_->cend(),
                                              this->outputIndices_, accumulatedSumVectorPtr_->cbegin(),
                                              accumulatedSumVectorPtr_->cend());
                        const IScoreVector& scoreVector = this->ruleEvaluationPtr_->calculateScores(
                          this->state_.statisticMatrixPtr->majorityLabelVectorPtr->cbegin(),
                          this->state_.statisticMatrixPtr->majorityLabelVectorPtr->cend(), this->totalSumVector_,
                          tmpVector_);
                        return this->state_.createUpdateCandidate(scoreVector);
                    }
            };

            const WeightVector& weights_;

            const RuleEvaluationFactory& ruleEvaluationFactory_;

            StatisticVector totalSumVector_;

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
                : AbstractStatisticsSpace<State>(state), weights_(weights),
                  ruleEvaluationFactory_(ruleEvaluationFactory),
                  totalSumVector_(state.statisticMatrixPtr->labelMatrix.numCols, true),
                  subsetSumVector_(state.statisticMatrixPtr->labelMatrix.numCols, true) {
                initializeStatisticVector(weights, state.statisticMatrixPtr->getView(), totalSumVector_);
                initializeStatisticVector(weights, state.statisticMatrixPtr->getView(), subsetSumVector_);
            }

            /**
             * @param statistics A reference to an object of type `WeightedStatistics` to be copied
             */
            WeightedStatistics(const WeightedStatistics& statistics)
                : AbstractStatisticsSpace<State>(statistics.state_), weights_(statistics.weights_),
                  ruleEvaluationFactory_(statistics.ruleEvaluationFactory_),
                  totalSumVector_(statistics.totalSumVector_), subsetSumVector_(statistics.subsetSumVector_) {}

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
                subsetSumVector_.clear();
            }

            /**
             * @see `IWeightedStatistics::addCoveredStatistic`
             */
            void addCoveredStatistic(uint32 statisticIndex) override {
                addStatisticInternally(weights_, this->state_.statisticMatrixPtr->getView(), subsetSumVector_,
                                       statisticIndex);
            }

            /**
             * @see `IWeightedStatistics::removeCoveredStatistic`
             */
            void removeCoveredStatistic(uint32 statisticIndex) override {
                removeStatisticInternally(weights_, this->state_.statisticMatrixPtr->getView(), subsetSumVector_,
                                          statisticIndex);
            }

            /**
             * @see `IWeightedStatistics::createSubset`
             */
            std::unique_ptr<IResettableStatisticsSubset> createSubset(
              const BinaryDokVector& excludedStatisticIndices,
              const CompleteIndexVector& outputIndices) const override {
                return std::make_unique<StatisticsSubset<CompleteIndexVector>>(*this, excludedStatisticIndices,
                                                                               outputIndices);
            }

            /**
             * @see `IWeightedStatistics::createSubset`
             */
            std::unique_ptr<IResettableStatisticsSubset> createSubset(
              const BinaryDokVector& excludedStatisticIndices, const PartialIndexVector& outputIndices) const override {
                return std::make_unique<StatisticsSubset<PartialIndexVector>>(*this, excludedStatisticIndices,
                                                                              outputIndices);
            }
    };

    /**
     * An abstract base class for all statistics that provide access to the elements of confusion matrices.
     *
     * @tparam State                    The type of the state of the training process
     * @tparam RuleEvaluationFactory    The type of the factory that allows to create instances of the class that is
     *                                  used for calculating the predictions of rules, as well as corresponding quality
     *                                  scores
     */
    template<typename State, typename RuleEvaluationFactory>
    class AbstractCoverageStatistics : public AbstractStatistics<State>,
                                       virtual public ICoverageStatistics {
        protected:

            /**
             * A pointer to an object of template type `RuleEvaluationFactory` that allows to create instances of the
             * class that should be used for calculating the predictions of rules, as well as their overall quality.
             */
            const RuleEvaluationFactory* ruleEvaluationFactory_;

        public:

            /**
             * @param statePtr              An unique pointer to an object of template type `State` that represents the
             *                              state of the training process and allows to update it
             * @param ruleEvaluationFactory A reference to an object of template type `RuleEvaluationFactory` that
             *                              allows to create instances of the class that should be used for calculating
             *                              the predictions of rules, as well as their overall quality
             */
            AbstractCoverageStatistics(std::unique_ptr<State> statePtr,
                                       const RuleEvaluationFactory& ruleEvaluationFactory)
                : AbstractStatistics<State>(std::move(statePtr)), ruleEvaluationFactory_(&ruleEvaluationFactory) {}

            /**
             * @see `ICoverageStatistics::getSumOfUncoveredWeights`
             */
            float64 getSumOfUncoveredWeights() const override final {
                return this->statePtr_->statisticMatrixPtr->coverageMatrixPtr->getSumOfUncoveredWeights();
            }

            /**
             * @see `IStatistics::evaluatePrediction`
             */
            float64 evaluatePrediction(uint32 statisticIndex) const override final {
                throw std::runtime_error("not implemented");
            }
    };

}
