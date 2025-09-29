/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/vector_sparse_array_binary.hpp"
#include "mlrl/seco/statistics/statistics.hpp"

#include <memory>
#include <utility>

namespace seco {

    static inline bool hasNonZeroWeightInternally(const EqualWeightVector& weights, uint32 statisticIndex) {
        return true;
    }

    template<typename WeightVector>
    static inline bool hasNonZeroWeightInternally(const WeightVector& weights, uint32 statisticIndex) {
        return !isEqualToZero(weights[statisticIndex]);
    }

    template<typename LabelMatrix, typename CoverageMatrix, typename StatisticVector, typename IndexVector>
    static inline void addStatisticToSubsetInternally(const EqualWeightVector& weights, const LabelMatrix& labelMatrix,
                                                      const BinarySparseArrayVector& majorityLabelVector,
                                                      const CoverageMatrix& coverageMatrix, StatisticVector& vector,
                                                      const IndexVector& outputIndices, uint32 statisticIndex) {
        vector.addToSubset(statisticIndex, labelMatrix, majorityLabelVector.cbegin(), majorityLabelVector.cend(),
                           coverageMatrix, outputIndices, 1);
    }

    template<typename WeightVector, typename LabelMatrix, typename CoverageMatrix, typename StatisticVector,
             typename IndexVector>
    static inline void addStatisticToSubsetInternally(const WeightVector& weights, const LabelMatrix& labelMatrix,
                                                      const BinarySparseArrayVector& majorityLabelVector,
                                                      const CoverageMatrix& coverageMatrix, StatisticVector& vector,
                                                      const IndexVector& outputIndices, uint32 statisticIndex) {
        typename WeightVector::weight_type weight = weights[statisticIndex];
        vector.addToSubset(statisticIndex, labelMatrix, majorityLabelVector.cbegin(), majorityLabelVector.cend(),
                           coverageMatrix, outputIndices, weight);
    }

    /**
     * An abstract base class for all subsets of confusion matrices.
     *
     * @tparam State                    The type of the state of the training process
     * @tparam StatisticVector          The type of the vector that is used to store the sums of statistics
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
    class AbstractStatisticsSubset : virtual public IStatisticsSubset {
        protected:

            /**
             * An object of template type `StatisticVector` that stores the sums of statistics.
             */
            StatisticVector sumVector_;

            /**
             * A reference to an object of template type `State` that represents the state of the training process.
             */
            State& state_;

            /**
             * A reference to an object of template type `StatisticVector` that stores the total sums of statistics.
             */
            const StatisticVector& totalSumVector_;

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
             * An unique pointer to an object of type `IRuleEvaluation` that is used for calculating the predictions of
             * rules, as well as their overall quality.
             */
            const std::unique_ptr<IRuleEvaluation<StatisticVector>> ruleEvaluationPtr_;

        public:

            /**
             * @param state                 A reference to an object of template type `State` that represents the state
             *                              of the training process
             * @param totalSumVector        A reference to an object of template type `StatisticVector` that stores the
             *                              total sums of statistics
             * @param ruleEvaluationFactory A reference to an object of template type `RuleEvaluationFactory` that
             *                              allows to create instances of the class that should be used for calculating
             *                              the predictions of rules, as well as their overall quality
             * @param weights               A reference to an object of template type `WeightVector` that provides
             *                              access to the weights of individual statistics
             * @param outputIndices         A reference to an object of template type `IndexVector` that provides access
             *                              to the indices of the outputs that are included in the subset
             */
            AbstractStatisticsSubset(State& state, const StatisticVector& totalSumVector,
                                     const RuleEvaluationFactory& ruleEvaluationFactory, const WeightVector& weights,
                                     const IndexVector& outputIndices)
                : sumVector_(outputIndices.getNumElements(), true), state_(state), totalSumVector_(totalSumVector),
                  weights_(weights), outputIndices_(outputIndices),
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
                addStatisticToSubsetInternally(
                  weights_, state_.statisticMatrixPtr->labelMatrix, *state_.statisticMatrixPtr->majorityLabelVectorPtr,
                  *state_.statisticMatrixPtr->coverageMatrixPtr, sumVector_, outputIndices_, statisticIndex);
            }

            /**
             * @see `IStatisticsSubset::calculateScores`
             */
            std::unique_ptr<IStatisticsUpdateCandidate> calculateScores() override final {
                const IScoreVector& scoreVector = ruleEvaluationPtr_->calculateScores(
                  state_.statisticMatrixPtr->majorityLabelVectorPtr->cbegin(),
                  state_.statisticMatrixPtr->majorityLabelVectorPtr->cend(), totalSumVector_, sumVector_);
                return state_.createUpdateCandidate(scoreVector);
            }
    };

    template<typename LabelMatrix, typename CoverageMatrix, typename StatisticVector>
    static inline void initializeStatisticVector(const EqualWeightVector& weights, const LabelMatrix& labelMatrix,
                                                 const BinarySparseArrayVector& majorityLabelVector,
                                                 const CoverageMatrix& coverageMatrix,
                                                 StatisticVector& statisticVector) {
        uint32 numStatistics = weights.getNumElements();

        for (uint32 i = 0; i < numStatistics; i++) {
            statisticVector.add(i, labelMatrix, majorityLabelVector.cbegin(), majorityLabelVector.cend(),
                                coverageMatrix, 1);
        }
    }

    template<typename WeightVector, typename LabelMatrix, typename CoverageMatrix, typename StatisticVector>
    static inline void initializeStatisticVector(const WeightVector& weights, const LabelMatrix& labelMatrix,
                                                 const BinarySparseArrayVector& majorityLabelVector,
                                                 const CoverageMatrix& coverageMatrix,
                                                 StatisticVector& statisticVector) {
        uint32 numStatistics = weights.getNumElements();

        for (uint32 i = 0; i < numStatistics; i++) {
            typename WeightVector::weight_type weight = weights[i];
            statisticVector.add(i, labelMatrix, majorityLabelVector.cbegin(), majorityLabelVector.cend(),
                                coverageMatrix, weight);
        }
    }

    /**
     * A subset of confusion matrices.
     *
     * @tparam State                    The type of the state of the training process
     * @tparam StatisticVector          The type of the vectors that are used to store statistics
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
    class StatisticsSubset final
        : public AbstractStatisticsSubset<State, StatisticVector, RuleEvaluationFactory, WeightVector, IndexVector> {
        private:

            const std::unique_ptr<StatisticVector> totalSumVectorPtr_;

        public:

            /**
             * @param totalSumVectorPtr     An unique pointer to an object of template type `StatisticVector` that
             *                              stores the total sums of statistics
             * @param state                 A reference to an object of template type `State` that represents the state
             *                              of the training process
             * @param ruleEvaluationFactory A reference to an object of template type `RuleEvaluationFactory` that
             *                              allows to create instances of the class that should be used for calculating
             *                              the predictions of rules, as well as their overall quality
             * @param weights               A reference to an object of template type `WeightVector` that provides
             *                              access to the weights of individual statistics
             * @param outputIndices         A reference to an object of template type `IndexVector` that provides access
             *                              to the indices of the outputs that are included in the subset
             */
            StatisticsSubset(std::unique_ptr<StatisticVector> totalSumVectorPtr, State& state,
                             const RuleEvaluationFactory& ruleEvaluationFactory, const WeightVector& weights,
                             const IndexVector& outputIndices)
                : AbstractStatisticsSubset<State, StatisticVector, RuleEvaluationFactory, WeightVector, IndexVector>(
                    state, *totalSumVectorPtr, ruleEvaluationFactory, weights, outputIndices),
                  totalSumVectorPtr_(std::move(totalSumVectorPtr)) {
                initializeStatisticVector(weights, state.statisticMatrixPtr->labelMatrix,
                                          *state.statisticMatrixPtr->majorityLabelVectorPtr,
                                          *state.statisticMatrixPtr->coverageMatrixPtr, *totalSumVectorPtr_);
            }
    };

    template<typename LabelMatrix, typename CoverageMatrix, typename StatisticVector>
    static inline void addStatisticInternally(const EqualWeightVector& weights, const LabelMatrix& labelMatrix,
                                              const BinarySparseArrayVector& majorityLabelVector,
                                              const CoverageMatrix& coverageMatrix, StatisticVector& vector,
                                              uint32 statisticIndex) {
        vector.add(statisticIndex, labelMatrix, majorityLabelVector.cbegin(), majorityLabelVector.cend(),
                   coverageMatrix, 1);
    }

    template<typename WeightVector, typename LabelMatrix, typename CoverageMatrix, typename StatisticVector>
    static inline void addStatisticInternally(const WeightVector& weights, const LabelMatrix& labelMatrix,
                                              const BinarySparseArrayVector& majorityLabelVector,
                                              const CoverageMatrix& coverageMatrix, StatisticVector& vector,
                                              uint32 statisticIndex) {
        typename WeightVector::weight_type weight = weights[statisticIndex];
        vector.add(statisticIndex, labelMatrix, majorityLabelVector.cbegin(), majorityLabelVector.cend(),
                   coverageMatrix, weight);
    }

    template<typename LabelMatrix, typename CoverageMatrix, typename StatisticVector>
    static inline void removeStatisticInternally(const EqualWeightVector& weights, const LabelMatrix& labelMatrix,
                                                 const BinarySparseArrayVector& majorityLabelVector,
                                                 const CoverageMatrix& coverageMatrix, StatisticVector& vector,
                                                 uint32 statisticIndex) {
        vector.remove(statisticIndex, labelMatrix, majorityLabelVector.cbegin(), majorityLabelVector.cend(),
                      coverageMatrix, 1);
    }

    template<typename WeightVector, typename LabelMatrix, typename CoverageMatrix, typename StatisticVector>
    static inline void removeStatisticInternally(const WeightVector& weights, const LabelMatrix& labelMatrix,
                                                 const BinarySparseArrayVector& majorityLabelVector,
                                                 const CoverageMatrix& coverageMatrix, StatisticVector& vector,
                                                 uint32 statisticIndex) {
        typename WeightVector::weight_type weight = weights[statisticIndex];
        vector.remove(statisticIndex, labelMatrix, majorityLabelVector.cbegin(), majorityLabelVector.cend(),
                      coverageMatrix, weight);
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
    class WeightedStatistics final : virtual public IWeightedStatistics {
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
                : virtual public IResettableStatisticsSubset,
                  public AbstractStatisticsSubset<State, StatisticVector, RuleEvaluationFactory, WeightVector,
                                                  IndexVector> {
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
                        : AbstractStatisticsSubset<State, StatisticVector, RuleEvaluationFactory, WeightVector,
                                                   IndexVector>(statistics.state_, statistics.totalSumVector_,
                                                                statistics.ruleEvaluationFactory_, statistics.weights_,
                                                                outputIndices),
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
                                removeStatisticInternally(this->weights_, this->state_.statisticMatrixPtr->labelMatrix,
                                                          *this->state_.statisticMatrixPtr->majorityLabelVectorPtr,
                                                          *this->state_.statisticMatrixPtr->coverageMatrixPtr,
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
                            accumulatedSumVectorPtr_->add(this->sumVector_.cbegin(), this->sumVector_.cend());
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

        protected:

            /**
             * A reference to an object of template type `State` that represents the state of the training process.
             */
            State& state_;

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
                : weights_(weights), ruleEvaluationFactory_(ruleEvaluationFactory),
                  totalSumVector_(state.statisticMatrixPtr->labelMatrix.numCols, true),
                  subsetSumVector_(state.statisticMatrixPtr->labelMatrix.numCols, true), state_(state) {
                initializeStatisticVector(weights, state_.statisticMatrixPtr->labelMatrix,
                                          *state_.statisticMatrixPtr->majorityLabelVectorPtr,
                                          *state.statisticMatrixPtr->coverageMatrixPtr, totalSumVector_);
                initializeStatisticVector(weights, state_.statisticMatrixPtr->labelMatrix,
                                          *state_.statisticMatrixPtr->majorityLabelVectorPtr,
                                          *state.statisticMatrixPtr->coverageMatrixPtr, subsetSumVector_);
            }

            /**
             * @param statistics A reference to an object of type `WeightedStatistics` to be copied
             */
            WeightedStatistics(const WeightedStatistics& statistics)
                : weights_(statistics.weights_), ruleEvaluationFactory_(statistics.ruleEvaluationFactory_),
                  totalSumVector_(statistics.totalSumVector_), subsetSumVector_(statistics.subsetSumVector_),
                  state_(statistics.state_) {}

            /**
             * @see `IStatisticsSpace::getNumStatistics`
             */
            uint32 getNumStatistics() const override {
                return state_.statisticMatrixPtr->getNumRows();
            }

            /**
             * @see `IStatisticsSpace::getNumOutputs`
             */
            uint32 getNumOutputs() const override {
                return state_.statisticMatrixPtr->getNumCols();
            }

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
                addStatisticInternally(weights_, state_.statisticMatrixPtr->labelMatrix,
                                       *state_.statisticMatrixPtr->majorityLabelVectorPtr,
                                       *state_.statisticMatrixPtr->coverageMatrixPtr, subsetSumVector_, statisticIndex);
            }

            /**
             * @see `IWeightedStatistics::removeCoveredStatistic`
             */
            void removeCoveredStatistic(uint32 statisticIndex) override {
                removeStatisticInternally(
                  weights_, state_.statisticMatrixPtr->labelMatrix, *state_.statisticMatrixPtr->majorityLabelVectorPtr,
                  *state_.statisticMatrixPtr->coverageMatrixPtr, subsetSumVector_, statisticIndex);
            }

            /**
             * @see `IStatisticsSpace::createSubset`
             */
            std::unique_ptr<IResettableStatisticsSubset> createSubset(
              const BinaryDokVector& excludedStatisticIndices,
              const CompleteIndexVector& outputIndices) const override {
                return std::make_unique<StatisticsSubset<CompleteIndexVector>>(*this, excludedStatisticIndices,
                                                                               outputIndices);
            }

            /**
             * @see `IStatisticsSpace::createSubset`
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
    class AbstractStatistics : virtual public ICoverageStatistics {
        protected:

            /**
             * An unique pointer to the state of the training process.
             */
            const std::unique_ptr<State> statePtr_;

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
            AbstractStatistics(std::unique_ptr<State> statePtr, const RuleEvaluationFactory& ruleEvaluationFactory)
                : statePtr_(std::move(statePtr)), ruleEvaluationFactory_(&ruleEvaluationFactory) {}

            /**
             * @see `ICoverageStatistics::getSumOfUncoveredWeights`
             */
            float64 getSumOfUncoveredWeights() const override final {
                return statePtr_->statisticMatrixPtr->coverageMatrixPtr->getSumOfUncoveredWeights();
            }

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
                throw std::runtime_error("not implemented");
            }
    };

}
