/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/statistics/statistics.hpp"
#include "statistics_update_candidate.hpp"

#include <memory>
#include <utility>

namespace boosting {

    static inline bool hasNonZeroWeightInternally(const EqualWeightVector& weights, uint32 statisticIndex) {
        return true;
    }

    template<typename WeightVector>
    static inline bool hasNonZeroWeightInternally(const WeightVector& weights, uint32 statisticIndex) {
        return !isEqualToZero(weights[statisticIndex]);
    }

    template<typename StatisticView, typename StatisticVector, typename IndexVector>
    static inline void addStatisticToSubsetInternally(const EqualWeightVector& weights,
                                                      const StatisticView& statisticView, StatisticVector& vector,
                                                      const IndexVector& outputIndices, uint32 statisticIndex) {
        vector.addToSubset(statisticView, statisticIndex, outputIndices);
    }

    template<typename WeightVector, typename StatisticView, typename StatisticVector, typename IndexVector>
    static inline void addStatisticToSubsetInternally(const WeightVector& weights, const StatisticView& statisticView,
                                                      StatisticVector& vector, const IndexVector& outputIndices,
                                                      uint32 statisticIndex) {
        typename WeightVector::weight_type weight = weights[statisticIndex];
        vector.addToSubset(statisticView, statisticIndex, outputIndices, weight);
    }

    /**
     * A subset of gradients and Hessians that are calculated according to a loss function and are accessible via a
     * view.
     *
     * @tparam State                    The type of the state of the boosting process
     * @tparam StatisticVector          The type of the vector that is used to store the sums of gradients and Hessians
     * @tparam RuleEvaluationFactory    The type of the factory that allows to create instances of the class that is
     *                                  used for calculating the predictions of rules, as well as corresponding quality
     *                                  scores
     * @tparam WeightVector             The type of the vector that provides access to the weights of individual
     *                                  statistics
     * @tparam IndexVector              The type of the vector that provides access to the indices of the outputs that
     *                                  are included in the subset
     */
    template<typename State, typename StatisticVector, typename RuleEvaluationFactory, typename WeightVector,
             typename IndexVector>
    class StatisticsSubset : virtual public IStatisticsSubset {
        protected:

            /**
             * An object of template type `StatisticVector` that stores the sums of gradients and Hessians.
             */
            StatisticVector sumVector_;

            /**
             * A reference to an object of template type `State` that represents the state of the boosting process.
             */
            State& state_;

            /**
             * A reference to an object of template type `WeightVector` that provides access to the weights of
             * individual statistics.
             */
            const WeightVector& weights_;

            /**
             * A reference to an object of template type `IndexVector` that provides access to the indices of the
             * outputs that are included in the subset.
             */
            const IndexVector& outputIndices_;

            /**
             * An unique pointer to an object of type `IRuleEvaluation` that is used to calculate the predictions of
             * rules, as well as their overall quality.
             */
            const std::unique_ptr<IRuleEvaluation<StatisticVector>> ruleEvaluationPtr_;

        public:

            /**
             * @param state                 A reference to an object of template type `State` that represents the state
             *                              of the boosting process
             * @param ruleEvaluationFactory A reference to an object of template type `RuleEvaluationFactory` that
             *                              allows to create instances of the class that is used for calculating the
             *                              predictions of rules, as well as their overall quality
             * @param weights               A reference to an object of template type `WeightVector` that provides
             *                              access to the weights of individual statistics
             * @param outputIndices         A reference to an object of template type `IndexVector` that provides access
             *                              to the indices of the outputs that are included in the subset
             */
            StatisticsSubset(State& state, const RuleEvaluationFactory& ruleEvaluationFactory,
                             const WeightVector& weights, const IndexVector& outputIndices)
                : sumVector_(outputIndices.getNumElements(), true), state_(state), weights_(weights),
                  outputIndices_(outputIndices),
                  ruleEvaluationPtr_(ruleEvaluationFactory.create(sumVector_, outputIndices)) {}

            /**
             * @see `IStatisticsSubset::hasNonZeroWeight`
             */
            bool hasNonZeroWeight(uint32 statisticIndex) const override final {
                return hasNonZeroWeightInternally(weights_, statisticIndex);
            }

            /**
             * @see `IStatisticsSubset::addToSubset`
             */
            void addToSubset(uint32 statisticIndex) override final {
                addStatisticToSubsetInternally(weights_, state_.statisticMatrixPtr->getView(), sumVector_,
                                               outputIndices_, statisticIndex);
            }

            /**
             * @see `IStatisticsSubset::calculateScores`
             */
            std::unique_ptr<StatisticsUpdateCandidate> calculateScores() override final {
                const IScoreVector& scoreVector = ruleEvaluationPtr_->calculateScores(sumVector_);
                return std::make_unique<BoostingStatisticsUpdateCandidate<State>>(state_, scoreVector);
            }
    };

    /**
     * An abstract base class for all statistics that provide access to gradients and Hessians that are calculated
     * according to a loss function.
     *
     * @tparam State                    The type of the state of the boosting process
     * @tparam StatisticVector          The type of the vectors that are used to store gradients and Hessians
     * @tparam RuleEvaluationFactory    The type of the factory that allows to create instances of the class that is
     *                                  used for calculating the predictions of rules, as well as corresponding quality
     *                                  scores
     * @tparam WeightVector             The type of the vector that provides access to the weights of individual
     *                                  statistics
     */
    template<typename State, typename StatisticVector, typename RuleEvaluationFactory, typename WeightVector>
    class AbstractStatisticsSpace : virtual public IStatisticsSpace {
        protected:

            /**
             * An abstract base class for all subsets of the gradients and Hessians that are stored by an instance of
             * the class `AbstractStatisticsSpace`.
             *
             * @tparam IndexVector The type of the vector that provides access to the indices of the outputs that are
             *                     included in the subset
             */
            template<typename IndexVector>
            class AbstractStatisticsSubset
                : public StatisticsSubset<State, StatisticVector, RuleEvaluationFactory, WeightVector, IndexVector>,
                  virtual public IResettableStatisticsSubset {
                private:

                    StatisticVector tmpVector_;

                    std::unique_ptr<StatisticVector> accumulatedSumVectorPtr_;

                protected:

                    /**
                     * A pointer to an object of template type `StatisticVector` that stores the total sum of all
                     * gradients and Hessians.
                     */
                    const StatisticVector* totalSumVector_;

                public:

                    /**
                     * @param statistics        A reference to an object of type `AbstractStatisticsSpace` that stores
                     *                          the gradients and Hessians
                     * @param totalSumVector    A reference to an object of template type `StatisticVector` that stores
                     *                          the total sums of gradients and Hessians
                     * @param outputIndices     A reference to an object of template type `IndexVector` that provides
                     *                          access to the indices of the outputs that are included in the subset
                     */
                    AbstractStatisticsSubset(const AbstractStatisticsSpace& statistics,
                                             const StatisticVector& totalSumVector, const IndexVector& outputIndices)
                        : StatisticsSubset<State, StatisticVector, RuleEvaluationFactory, WeightVector, IndexVector>(
                            statistics.state_, statistics.ruleEvaluationFactory_, statistics.weights_, outputIndices),
                          tmpVector_(outputIndices.getNumElements()), totalSumVector_(&totalSumVector) {}

                    /**
                     * @see `IResettableStatisticsSubset::resetSubset`
                     */
                    void resetSubset() override final {
                        if (!accumulatedSumVectorPtr_) {
                            // Create a vector for storing the accumulated sums of gradients and Hessians, if
                            // necessary...
                            accumulatedSumVectorPtr_ = std::make_unique<StatisticVector>(this->sumVector_);
                        } else {
                            // Add the sums of gradients and Hessians to the accumulated sums of gradients and
                            // Hessians...
                            accumulatedSumVectorPtr_->add(this->sumVector_);
                        }

                        // Reset the sums of gradients and Hessians to zero...
                        this->sumVector_.clear();
                    }

                    /**
                     * @see `IResettableStatisticsSubset::calculateScoresAccumulated`
                     */
                    std::unique_ptr<StatisticsUpdateCandidate> calculateScoresAccumulated() override final {
                        const IScoreVector& scoreVector =
                          this->ruleEvaluationPtr_->calculateScores(*accumulatedSumVectorPtr_);
                        return std::make_unique<BoostingStatisticsUpdateCandidate<State>>(this->state_, scoreVector);
                    }

                    /**
                     * @see `IResettableStatisticsSubset::calculateScoresUncovered`
                     */
                    std::unique_ptr<StatisticsUpdateCandidate> calculateScoresUncovered() override final {
                        tmpVector_.difference(*totalSumVector_, this->outputIndices_, this->sumVector_);
                        const IScoreVector& scoreVector = this->ruleEvaluationPtr_->calculateScores(tmpVector_);
                        return std::make_unique<BoostingStatisticsUpdateCandidate<State>>(this->state_, scoreVector);
                    }

                    /**
                     * @see `IResettableStatisticsSubset::calculateScoresUncoveredAccumulated`
                     */
                    std::unique_ptr<StatisticsUpdateCandidate> calculateScoresUncoveredAccumulated() override final {
                        tmpVector_.difference(*totalSumVector_, this->outputIndices_, *accumulatedSumVectorPtr_);
                        const IScoreVector& scoreVector = this->ruleEvaluationPtr_->calculateScores(tmpVector_);
                        return std::make_unique<BoostingStatisticsUpdateCandidate<State>>(this->state_, scoreVector);
                    }
            };

        protected:

            /**
             * A reference to an object of template type `State` that represents the state of the boosting process.
             */
            State& state_;

            /**
             * A reference to an object of template type `RuleEvaluationFactory` that is used to create instances of the
             * class that is used for calculating the predictions of rules, as well as their overall quality.
             */
            const RuleEvaluationFactory& ruleEvaluationFactory_;

            /**
             * A reference to an object of template type `WeightVector` that provides access to the weights of
             * individual statistics.
             */
            const WeightVector& weights_;

        public:

            /**
             * @param state                 A reference to an object of template type `State` that represents the state
             *                              of the boosting process
             * @param ruleEvaluationFactory A reference to an object of template type `RuleEvaluationFactory` that
             *                              allows to create instances of the class that should be used for calculating
             *                              the predictions of rules, as well as their overall quality
             * @param weights               A reference to an object of template type `WeightVector` that provides
             *                              access to the weights of individual statistics
             */
            AbstractStatisticsSpace(State& state, const RuleEvaluationFactory& ruleEvaluationFactory,
                                    const WeightVector& weights)
                : state_(state), ruleEvaluationFactory_(ruleEvaluationFactory), weights_(weights) {}

            /**
             * @see `IStatisticsSpace::getNumStatistics`
             */
            uint32 getNumStatistics() const override final {
                return state_.statisticMatrixPtr->getNumRows();
            }

            /**
             * @see `IStatisticsSpace::getNumOutputs`
             */
            uint32 getNumOutputs() const override final {
                return state_.statisticMatrixPtr->getNumCols();
            }
    };

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
     * @tparam State                    The type of the state of the boosting process
     * @tparam StatisticVector          The type of the vectors that are used to store gradients and Hessians
     * @tparam RuleEvaluationFactory    The type of the factory that allows to create instances of the class that is
     *                                  used for calculating the predictions of rules, as well as corresponding quality
     *                                  scores
     * @tparam WeightVector             The type of the vector that provides access to the weights of individual
     *                                  statistics
     */
    template<typename State, typename StatisticVector, typename RuleEvaluationFactory, typename WeightVector>
    class WeightedStatistics final
        : virtual public IWeightedStatistics,
          public AbstractStatisticsSpace<State, StatisticVector, RuleEvaluationFactory, WeightVector> {
        private:

            /**
             * Provides access to a subset of the gradients and Hessians that are stored by an instance of the class
             * `WeightedStatistics`.
             *
             * @tparam IndexVector The type of the vector that provides access to the indices of the outputs that are
             *                     included in the subset
             */
            template<typename IndexVector>
            class StatisticsSubset final
                : public AbstractStatisticsSpace<State, StatisticVector, RuleEvaluationFactory,
                                                 WeightVector>::template AbstractStatisticsSubset<IndexVector> {
                private:

                    std::unique_ptr<StatisticVector> totalCoverableSumVectorPtr_;

                public:

                    /**
                     * @param statistics                A reference to an object of type `WeightedStatistics` that
                     *                                  stores the gradients and Hessians
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
                        : AbstractStatisticsSpace<State, StatisticVector, RuleEvaluationFactory, WeightVector>::
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
             *                              of the boosting process
             * @param ruleEvaluationFactory A reference to an object of template type `RuleEvaluationFactory` that
             *                              allows to create instances of the class that should be used for calculating
             *                              the predictions of rules, as well as their overall quality
             * @param weights               A reference to an object of template type `WeightVector` that provides
             *                              access to the weights of individual statistics
             */
            WeightedStatistics(State& state, const RuleEvaluationFactory& ruleEvaluationFactory,
                               const WeightVector& weights)
                : AbstractStatisticsSpace<State, StatisticVector, RuleEvaluationFactory, WeightVector>(
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
                : AbstractStatisticsSpace<State, StatisticVector, RuleEvaluationFactory, WeightVector>(
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

    /**
     * An abstract base class for all statistics that provide access to gradients and Hessians that are calculated
     * according to a loss function.
     *
     * @tparam State                    The type of the state of the boosting process
     * @tparam EvaluationMeasure        The type of the evaluation measure that is used to assess the quality of
     *                                  predictions for a specific statistic
     * @tparam RuleEvaluationFactory    The type of the factory that allows to create instances of the class that is
     *                                  used for calculating the predictions of rules, as well as corresponding quality
     *                                  scores
     */
    template<typename State, typename EvaluationMeasure, typename RuleEvaluationFactory>
    class AbstractStatistics : virtual public IBoostingStatistics {
        private:

            /**
             * Allows updating the state of statistics based on the predictions of a rule.
             *
             * @tparam Prediction The type of the predictions
             */
            template<typename Prediction>
            class Update final : public IStatisticsUpdate {
                private:

                    State& state_;

                    const Prediction& prediction_;

                public:

                    /**
                     * @param state         A reference to an object of template type `State` that should be updated
                     * @param prediction    The predictions of the rule
                     */
                    Update(State& state, const Prediction& prediction) : state_(state), prediction_(prediction) {}

                    void applyPrediction(uint32 statisticIndex) override {
                        state_.update(statisticIndex, prediction_.values_cbegin(), prediction_.values_cend(),
                                      prediction_.indices_cbegin(), prediction_.indices_cend());
                    }

                    void revertPrediction(uint32 statisticIndex) override {
                        state_.revert(statisticIndex, prediction_.values_cbegin(), prediction_.values_cend(),
                                      prediction_.indices_cbegin(), prediction_.indices_cend());
                    }
            };

        protected:

            /**
             * An unique pointer to the state of the boosting process.
             */
            const std::unique_ptr<State> statePtr_;

            /**
             * An unique pointer to the evaluation measure that should be used to assess the quality of predictions for
             * a specific statistic.
             */
            std::unique_ptr<EvaluationMeasure> evaluationMeasurePtr_;

            /**
             * A pointer to an object of template type `RuleEvaluationFactory` that allows to create instances of the
             * class that should be used for calculating the predictions of rules, as well as their overall quality.
             */
            const RuleEvaluationFactory* ruleEvaluationFactory_;

        public:

            /**
             * @param statePtr              An unique pointer to an object of template type `State` that represents the
             *                              state of the boosting process and allows to update it
             * @param evaluationMeasurePtr  An unique pointer to an object of template type `EvaluationMeasure` that
             *                              implements the evaluation measure that should be used to assess the quality
             *                              of predictions for a specific statistic
             * @param ruleEvaluationFactory A reference to an object of template type `RuleEvaluationFactory` that
             *                              allows to create instances of the class that should be used for calculating
             *                              the predictions of rules, as well as their overall quality
             */
            AbstractStatistics(std::unique_ptr<State> statePtr, std::unique_ptr<EvaluationMeasure> evaluationMeasurePtr,
                               const RuleEvaluationFactory& ruleEvaluationFactory)
                : statePtr_(std::move(statePtr)), evaluationMeasurePtr_(std::move(evaluationMeasurePtr)),
                  ruleEvaluationFactory_(&ruleEvaluationFactory) {}

            /**
             * @see `IStatistics::getNumStatistics`
             */
            uint32 getNumStatistics() const override final {
                return statePtr_->statisticMatrixPtr->getNumRows();
            }

            /**
             * @see `IStatistics::getNumOutputs`
             */
            uint32 getNumOutputs() const override final {
                return statePtr_->statisticMatrixPtr->getNumCols();
            }

            /**
             * @see `IStatistics::evaluatePrediction`
             */
            float64 evaluatePrediction(uint32 statisticIndex) const override final {
                return evaluationMeasurePtr_->evaluate(statisticIndex, statePtr_->outputMatrix,
                                                       statePtr_->scoreMatrixPtr->getView());
            }
    };

}
