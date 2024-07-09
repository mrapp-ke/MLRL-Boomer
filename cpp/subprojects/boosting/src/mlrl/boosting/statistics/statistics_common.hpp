/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/statistics/statistics.hpp"

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
        float64 weight = weights[statisticIndex];
        vector.addToSubset(statisticView, statisticIndex, outputIndices, weight);
    }

    /**
     * A subset of gradients and Hessians that are calculated according to a loss function and are accessible via a
     * view.
     *
     * @tparam StatisticVector          The type of the vector that is used to store the sums of gradients and Hessians
     * @tparam StatisticView            The type of the view that provides access to the gradients and Hessians
     * @tparam RuleEvaluationFactory    The type of the factory that allows to create instances of the class that is
     *                                  used for calculating the predictions of rules, as well as corresponding quality
     *                                  scores
     * @tparam WeightVector             The type of the vector that provides access to the weights of individual
     *                                  statistics
     * @tparam IndexVector              The type of the vector that provides access to the indices of the outputs that
     *                                  are included in the subset
     */
    template<typename StatisticVector, typename StatisticView, typename RuleEvaluationFactory, typename WeightVector,
             typename IndexVector>
    class StatisticsSubset : virtual public IStatisticsSubset {
        protected:

            /**
             * An object of template type `StatisticVector` that stores the sums of gradients and Hessians.
             */
            StatisticVector sumVector_;

            /**
             * A reference to an object of template type `StatisticView` that provides access to the gradients and
             * Hessians.
             */
            const StatisticView& statisticView_;

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
             * @param statisticView         A reference to an object of template type `StatisticView` that provides
             *                              access to the gradients and Hessians
             * @param ruleEvaluationFactory A reference to an object of template type `RuleEvaluationFactory` that
             *                              allows to create instances of the class that is used for calculating the
             *                              predictions of rules, as well as their overall quality
             * @param weights               A reference to an object of template type `WeightVector` that provides
             *                              access to the weights of individual statistics
             * @param outputIndices         A reference to an object of template type `IndexVector` that provides access
             *                              to the indices of the outputs that are included in the subset
             */
            StatisticsSubset(const StatisticView& statisticView, const RuleEvaluationFactory& ruleEvaluationFactory,
                             const WeightVector& weights, const IndexVector& outputIndices)
                : sumVector_(outputIndices.getNumElements(), true), statisticView_(statisticView), weights_(weights),
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
                addStatisticToSubsetInternally(weights_, statisticView_, sumVector_, outputIndices_, statisticIndex);
            }

            /**
             * @see `IStatisticsSubset::calculateScores`
             */
            const IScoreVector& calculateScores() override final {
                return ruleEvaluationPtr_->calculateScores(sumVector_);
            }
    };

    /**
     * An abstract base class for all statistics that provide access to gradients and Hessians that are calculated
     * according to a loss function.
     *
     * @tparam StatisticVector          The type of the vectors that are used to store gradients and Hessians
     * @tparam StatisticView            The type of the view that provides access to the gradients and Hessians
     * @tparam RuleEvaluationFactory    The type of the factory that allows to create instances of the class that is
     *                                  used for calculating the predictions of rules, as well as corresponding quality
     *                                  scores
     * @tparam WeightVector             The type of the vector that provides access to the weights of individual
     *                                  statistics
     */
    template<typename StatisticVector, typename StatisticView, typename RuleEvaluationFactory, typename WeightVector>
    class AbstractImmutableWeightedStatistics : virtual public IImmutableWeightedStatistics {
        protected:

            /**
             * An abstract base class for all subsets of the gradients and Hessians that are stored by an instance of
             * the class `AbstractImmutableWeightedStatistics`.
             *
             * @tparam IndexVector The type of the vector that provides access to the indices of the outputs that are
             *                     included in the subset
             */
            template<typename IndexVector>
            class AbstractWeightedStatisticsSubset
                : public StatisticsSubset<StatisticVector, StatisticView, RuleEvaluationFactory, WeightVector,
                                          IndexVector>,
                  virtual public IWeightedStatisticsSubset {
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
                     * @param statistics        A reference to an object of type `AbstractImmutableWeightedStatistics`
                     *                          that stores the gradients and Hessians
                     * @param totalSumVector    A reference to an object of template type `StatisticVector` that stores
                     *                          the total sums of gradients and Hessians
                     * @param outputIndices     A reference to an object of template type `IndexVector` that provides
                     *                          access to the indices of the outputs that are included in the subset
                     */
                    AbstractWeightedStatisticsSubset(const AbstractImmutableWeightedStatistics& statistics,
                                                     const StatisticVector& totalSumVector,
                                                     const IndexVector& outputIndices)
                        : StatisticsSubset<StatisticVector, StatisticView, RuleEvaluationFactory, WeightVector,
                                           IndexVector>(statistics.statisticView_, statistics.ruleEvaluationFactory_,
                                                        statistics.weights_, outputIndices),
                          tmpVector_(outputIndices.getNumElements()), totalSumVector_(&totalSumVector) {}

                    /**
                     * @see `IWeightedStatisticsSubset::resetSubset`
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
                     * @see `IWeightedStatisticsSubset::calculateScoresAccumulated`
                     */
                    const IScoreVector& calculateScoresAccumulated() override final {
                        return this->ruleEvaluationPtr_->calculateScores(*accumulatedSumVectorPtr_);
                    }

                    /**
                     * @see `IWeightedStatisticsSubset::calculateScoresUncovered`
                     */
                    const IScoreVector& calculateScoresUncovered() override final {
                        tmpVector_.difference(*totalSumVector_, this->outputIndices_, this->sumVector_);
                        return this->ruleEvaluationPtr_->calculateScores(tmpVector_);
                    }

                    /**
                     * @see `IWeightedStatisticsSubset::calculateScoresUncoveredAccumulated`
                     */
                    const IScoreVector& calculateScoresUncoveredAccumulated() override final {
                        tmpVector_.difference(*totalSumVector_, this->outputIndices_, *accumulatedSumVectorPtr_);
                        return this->ruleEvaluationPtr_->calculateScores(tmpVector_);
                    }
            };

        protected:

            /**
             * A reference to an object of template type `StatisticView` that stores the gradients and Hessians.
             */
            const StatisticView& statisticView_;

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
             * @param statisticView         A reference to an object of template type `StatisticView` that provides
             *                              access to the gradients and Hessians
             * @param ruleEvaluationFactory A reference to an object of template type `RuleEvaluationFactory` that
             *                              allows to create instances of the class that should be used for calculating
             *                              the predictions of rules, as well as their overall quality
             * @param weights               A reference to an object of template type `WeightVector` that provides
             *                              access to the weights of individual statistics
             */
            AbstractImmutableWeightedStatistics(const StatisticView& statisticView,
                                                const RuleEvaluationFactory& ruleEvaluationFactory,
                                                const WeightVector& weights)
                : statisticView_(statisticView), ruleEvaluationFactory_(ruleEvaluationFactory), weights_(weights) {}

            /**
             * @see `IImmutableWeightedStatistics::getNumStatistics`
             */
            uint32 getNumStatistics() const override final {
                return statisticView_.numRows;
            }

            /**
             * @see `IImmutableWeightedStatistics::getNumOutputs`
             */
            uint32 getNumOutputs() const override final {
                return statisticView_.numCols;
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
        float64 weight = weights[statisticIndex];
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
        float64 weight = weights[statisticIndex];
        statisticVector.remove(statisticView, statisticIndex, weight);
    }

    /**
     * Provides access to weighted gradients and Hessians that are calculated according to a loss function and allows to
     * update the gradients and Hessians after a new rule has been learned.
     *
     * @tparam StatisticVector          The type of the vectors that are used to store gradients and Hessians
     * @tparam StatisticView            The type of the view that provides access to the gradients and Hessians
     * @tparam RuleEvaluationFactory    The type of the factory that allows to create instances of the class that is
     *                                  used for calculating the predictions of rules, as well as corresponding quality
     *                                  scores
     * @tparam WeightVector             The type of the vector that provides access to the weights of individual
     *                                  statistics
     */
    template<typename StatisticVector, typename StatisticView, typename RuleEvaluationFactory, typename WeightVector>
    class WeightedStatistics final : virtual public IWeightedStatistics,
                                     public AbstractImmutableWeightedStatistics<StatisticVector, StatisticView,
                                                                                RuleEvaluationFactory, WeightVector> {
        private:

            /**
             * Provides access to a subset of the gradients and Hessians that are stored by an instance of the class
             * `WeightedStatistics`.
             *
             * @tparam IndexVector The type of the vector that provides access to the indices of the outputs that are
             *                     included in the subset
             */
            template<typename IndexVector>
            class WeightedStatisticsSubset final
                : public AbstractImmutableWeightedStatistics<
                    StatisticVector, StatisticView, RuleEvaluationFactory,
                    WeightVector>::template AbstractWeightedStatisticsSubset<IndexVector> {
                private:

                    std::unique_ptr<StatisticVector> totalCoverableSumVectorPtr_;

                public:

                    /**
                     * @param statistics        A reference to an object of type `WeightedStatistics` that stores the
                     *                          gradients and Hessians
                     * @param totalSumVector    A reference to an object of template type `StatisticVector` that stores
                     *                          the total sums of gradients and Hessians
                     * @param outputIndices     A reference to an object of template type `IndexVector` that provides
                     *                          access to the indices of the outputs that are included in the subset
                     */
                    WeightedStatisticsSubset(const WeightedStatistics& statistics,
                                             const StatisticVector& totalSumVector, const IndexVector& outputIndices)
                        : AbstractImmutableWeightedStatistics<
                            StatisticVector, StatisticView, RuleEvaluationFactory,
                            WeightVector>::template AbstractWeightedStatisticsSubset<IndexVector>(statistics,
                                                                                                  totalSumVector,
                                                                                                  outputIndices) {}

                    /**
                     * @see `IWeightedStatisticsSubset::addToMissing`
                     */
                    void addToMissing(uint32 statisticIndex) override {
                        // Create a vector for storing the totals sums of gradients and Hessians, if necessary...
                        if (!totalCoverableSumVectorPtr_) {
                            totalCoverableSumVectorPtr_ = std::make_unique<StatisticVector>(*this->totalSumVector_);
                            this->totalSumVector_ = totalCoverableSumVectorPtr_.get();
                        }

                        // Subtract the gradients and Hessians of the example at the given index (weighted by the given
                        // weight) from the total sums of gradients and Hessians...
                        removeStatisticInternally(this->weights_, this->statisticView_, *totalCoverableSumVectorPtr_,
                                                  statisticIndex);
                    }
            };

            const std::unique_ptr<StatisticVector> totalSumVectorPtr_;

        public:

            /**
             * @param statisticView         A reference to an object of template type `StatisticView` that provides
             *                              access to the gradients and Hessians
             * @param ruleEvaluationFactory A reference to an object of template type `RuleEvaluationFactory` that
             *                              allows to create instances of the class that should be used for calculating
             *                              the predictions of rules, as well as their overall quality
             * @param weights               A reference to an object of template type `WeightVector` that provides
             *                              access to the weights of individual statistics
             */
            WeightedStatistics(const StatisticView& statisticView, const RuleEvaluationFactory& ruleEvaluationFactory,
                               const WeightVector& weights)
                : AbstractImmutableWeightedStatistics<StatisticVector, StatisticView, RuleEvaluationFactory,
                                                      WeightVector>(statisticView, ruleEvaluationFactory, weights),
                  totalSumVectorPtr_(std::make_unique<StatisticVector>(statisticView.numCols, true)) {
                uint32 numStatistics = weights.getNumElements();

                for (uint32 i = 0; i < numStatistics; i++) {
                    addStatisticInternally(weights, statisticView, *totalSumVectorPtr_, i);
                }
            }

            /**
             * @param statistics A reference to an object of type `WeightedStatistics` to be copied
             */
            WeightedStatistics(const WeightedStatistics& statistics)
                : AbstractImmutableWeightedStatistics<StatisticVector, StatisticView, RuleEvaluationFactory,
                                                      WeightVector>(
                    statistics.statisticView_, statistics.ruleEvaluationFactory_, statistics.weights_),
                  totalSumVectorPtr_(std::make_unique<StatisticVector>(*statistics.totalSumVectorPtr_)) {}

            /**
             * @see `IWeightedStatistics::copy`
             */
            std::unique_ptr<IWeightedStatistics> copy() const override {
                return std::make_unique<
                  WeightedStatistics<StatisticVector, StatisticView, RuleEvaluationFactory, WeightVector>>(*this);
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
                addStatisticInternally(this->weights_, this->statisticView_, *totalSumVectorPtr_, statisticIndex);
            }

            /**
             * @see `IWeightedStatistics::removeCoveredStatistic`
             */
            void removeCoveredStatistic(uint32 statisticIndex) override {
                removeStatisticInternally(this->weights_, this->statisticView_, *totalSumVectorPtr_, statisticIndex);
            }

            /**
             * @see `IImmutableWeightedStatistics::createSubset`
             */
            std::unique_ptr<IWeightedStatisticsSubset> createSubset(
              const CompleteIndexVector& outputIndices) const override {
                return std::make_unique<WeightedStatisticsSubset<CompleteIndexVector>>(*this, *totalSumVectorPtr_,
                                                                                       outputIndices);
            }

            /**
             * @see `IImmutableWeightedStatistics::createSubset`
             */
            std::unique_ptr<IWeightedStatisticsSubset> createSubset(
              const PartialIndexVector& outputIndices) const override {
                return std::make_unique<WeightedStatisticsSubset<PartialIndexVector>>(*this, *totalSumVectorPtr_,
                                                                                      outputIndices);
            }
    };

    template<typename Prediction, typename ScoreMatrix>
    static inline void applyPredictionInternally(uint32 statisticIndex, const Prediction& prediction,
                                                 ScoreMatrix& scoreMatrix) {
        scoreMatrix.addToRowFromSubset(statisticIndex, prediction.values_cbegin(), prediction.values_cend(),
                                       prediction.indices_cbegin(), prediction.indices_cend());
    }

    template<typename Prediction, typename ScoreMatrix>
    static inline void revertPredictionInternally(uint32 statisticIndex, const Prediction& prediction,
                                                  ScoreMatrix& scoreMatrix) {
        scoreMatrix.removeFromRowFromSubset(statisticIndex, prediction.values_cbegin(), prediction.values_cend(),
                                            prediction.indices_cbegin(), prediction.indices_cend());
    }

    /**
     * An abstract base class for all statistics that provide access to gradients and Hessians that are calculated
     * according to a loss function.
     *
     * @tparam OutputMatrix             The type of the matrix that provides access to the ground truth of the training
     *                                  examples
     * @tparam StatisticVector          The type of the vectors that are used to store gradients and Hessians
     * @tparam StatisticMatrix          The type of the matrix that provides access to the gradients and Hessians
     * @tparam ScoreMatrix              The type of the matrices that are used to store predicted scores
     * @tparam LossFunction             The type of the loss function that is used to calculate gradients and Hessians
     * @tparam EvaluationMeasure        The type of the evaluation measure that is used to assess the quality of
     *                                  predictions for a specific statistic
     * @tparam RuleEvaluationFactory    The type of the factory that allows to create instances of the class that is
     *                                  used for calculating the predictions of rules, as well as corresponding quality
     *                                  scores
     */
    template<typename OutputMatrix, typename StatisticVector, typename StatisticMatrix, typename ScoreMatrix,
             typename LossFunction, typename EvaluationMeasure, typename RuleEvaluationFactory>
    class AbstractStatistics : virtual public IBoostingStatistics {
        protected:

            /**
             * An unique pointer to the loss function that should be used for calculating gradients and Hessians.
             */
            std::unique_ptr<LossFunction> lossPtr_;

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

            /**
             * The output matrix that provides access to the ground truth of the training examples.
             */
            const OutputMatrix& outputMatrix_;

            /**
             * An unique pointer to an object of template type `StatisticMatrix` that stores the gradients and Hessians.
             */
            const std::unique_ptr<StatisticMatrix> statisticMatrixPtr_;

            /**
             * An unique pointer to an object of template type `ScoreMatrix` that stores the currently predicted scores.
             */
            std::unique_ptr<ScoreMatrix> scoreMatrixPtr_;

            /**
             * Must be implemented by subclasses in order to update the statistics for all available outputs at a
             * specific index.
             */
            virtual void updateStatistics(uint32 statisticIndex, const CompletePrediction& prediction) = 0;

            /**
             * Must be implemented by subclasses in order to update the statistics for a subset of the available outputs
             * at a specific index.
             */
            virtual void updateStatistics(uint32 statisticIndex, const PartialPrediction& prediction) = 0;

        public:

            /**
             * @param lossPtr               An unique pointer to an object of template type `LossFunction` that
             *                              implements the loss function that should be used for calculating gradients
             *                              and Hessians
             * @param evaluationMeasurePtr  An unique pointer to an object of template type `EvaluationMeasure` that
             *                              implements the evaluation measure that should be used to assess the quality
             *                              of predictions for a specific statistic
             * @param ruleEvaluationFactory A reference to an object of template type `RuleEvaluationFactory` that
             *                              allows to create instances of the class that should be used for calculating
             *                              the predictions of rules, as well as their overall quality
             * @param outputMatrix          A reference to an object of template type `OutputMatrix` that provides
             *                              access to the ground truth of the training examples
             * @param statisticMatrixPtr    An unique pointer to an object of template type `StatisticMatrix` that
             *                              provides access to the gradients and Hessians
             * @param scoreMatrixPtr        An unique pointer to an object of template type `ScoreMatrix` that stores
             *                              the currently predicted scores
             */
            AbstractStatistics(std::unique_ptr<LossFunction> lossPtr,
                               std::unique_ptr<EvaluationMeasure> evaluationMeasurePtr,
                               const RuleEvaluationFactory& ruleEvaluationFactory, const OutputMatrix& outputMatrix,
                               std::unique_ptr<StatisticMatrix> statisticMatrixPtr,
                               std::unique_ptr<ScoreMatrix> scoreMatrixPtr)
                : lossPtr_(std::move(lossPtr)), evaluationMeasurePtr_(std::move(evaluationMeasurePtr)),
                  ruleEvaluationFactory_(&ruleEvaluationFactory), outputMatrix_(outputMatrix),
                  statisticMatrixPtr_(std::move(statisticMatrixPtr)), scoreMatrixPtr_(std::move(scoreMatrixPtr)) {}

            /**
             * @see `IStatistics::getNumStatistics`
             */
            uint32 getNumStatistics() const override final {
                return statisticMatrixPtr_->getNumRows();
            }

            /**
             * @see `IStatistics::getNumOutputs`
             */
            uint32 getNumOutputs() const override final {
                return statisticMatrixPtr_->getNumCols();
            }

            /**
             * @see `IStatistics::applyPrediction`
             */
            void applyPrediction(uint32 statisticIndex, const CompletePrediction& prediction) override final {
                applyPredictionInternally(statisticIndex, prediction, *scoreMatrixPtr_);
                this->updateStatistics(statisticIndex, prediction);
            }

            /**
             * @see `IStatistics::applyPrediction`
             */
            void applyPrediction(uint32 statisticIndex, const PartialPrediction& prediction) override final {
                applyPredictionInternally(statisticIndex, prediction, *scoreMatrixPtr_);
                this->updateStatistics(statisticIndex, prediction);
            }

            /**
             * @see `IStatistics::revertPrediction`
             */
            void revertPrediction(uint32 statisticIndex, const CompletePrediction& prediction) override final {
                revertPredictionInternally(statisticIndex, prediction, *scoreMatrixPtr_);
                this->updateStatistics(statisticIndex, prediction);
            }

            /**
             * @see `IStatistics::revertPrediction`
             */
            void revertPrediction(uint32 statisticIndex, const PartialPrediction& prediction) override final {
                revertPredictionInternally(statisticIndex, prediction, *scoreMatrixPtr_);
                this->updateStatistics(statisticIndex, prediction);
            }

            /**
             * @see `IStatistics::evaluatePrediction`
             */
            float64 evaluatePrediction(uint32 statisticIndex) const override final {
                return evaluationMeasurePtr_->evaluate(statisticIndex, outputMatrix_, scoreMatrixPtr_->getView());
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const CompleteIndexVector& outputIndices,
                                                            const EqualWeightVector& weights) const override final {
                return std::make_unique<
                  StatisticsSubset<StatisticVector, typename StatisticMatrix::view_type, RuleEvaluationFactory,
                                   EqualWeightVector, CompleteIndexVector>>(
                  statisticMatrixPtr_->getView(), *ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& outputIndices,
                                                            const EqualWeightVector& weights) const override final {
                return std::make_unique<StatisticsSubset<StatisticVector, typename StatisticMatrix::view_type,
                                                         RuleEvaluationFactory, EqualWeightVector, PartialIndexVector>>(
                  statisticMatrixPtr_->getView(), *ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const CompleteIndexVector& outputIndices,
                                                            const BitWeightVector& weights) const override final {
                return std::make_unique<StatisticsSubset<StatisticVector, typename StatisticMatrix::view_type,
                                                         RuleEvaluationFactory, BitWeightVector, CompleteIndexVector>>(
                  statisticMatrixPtr_->getView(), *ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& outputIndices,
                                                            const BitWeightVector& weights) const override final {
                return std::make_unique<StatisticsSubset<StatisticVector, typename StatisticMatrix::view_type,
                                                         RuleEvaluationFactory, BitWeightVector, PartialIndexVector>>(
                  statisticMatrixPtr_->getView(), *ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& outputIndices, const DenseWeightVector<uint32>& weights) const override final {
                return std::make_unique<
                  StatisticsSubset<StatisticVector, typename StatisticMatrix::view_type, RuleEvaluationFactory,
                                   DenseWeightVector<uint32>, CompleteIndexVector>>(
                  statisticMatrixPtr_->getView(), *ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const PartialIndexVector& outputIndices, const DenseWeightVector<uint32>& weights) const override final {
                return std::make_unique<
                  StatisticsSubset<StatisticVector, typename StatisticMatrix::view_type, RuleEvaluationFactory,
                                   DenseWeightVector<uint32>, PartialIndexVector>>(
                  statisticMatrixPtr_->getView(), *ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& outputIndices,
              const OutOfSampleWeightVector<EqualWeightVector>& weights) const override final {
                return std::make_unique<
                  StatisticsSubset<StatisticVector, typename StatisticMatrix::view_type, RuleEvaluationFactory,
                                   OutOfSampleWeightVector<EqualWeightVector>, CompleteIndexVector>>(
                  statisticMatrixPtr_->getView(), *ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const PartialIndexVector& outputIndices,
              const OutOfSampleWeightVector<EqualWeightVector>& weights) const override final {
                return std::make_unique<
                  StatisticsSubset<StatisticVector, typename StatisticMatrix::view_type, RuleEvaluationFactory,
                                   OutOfSampleWeightVector<EqualWeightVector>, PartialIndexVector>>(
                  statisticMatrixPtr_->getView(), *ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& outputIndices,
              const OutOfSampleWeightVector<BitWeightVector>& weights) const override final {
                return std::make_unique<
                  StatisticsSubset<StatisticVector, typename StatisticMatrix::view_type, RuleEvaluationFactory,
                                   OutOfSampleWeightVector<BitWeightVector>, CompleteIndexVector>>(
                  statisticMatrixPtr_->getView(), *ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const PartialIndexVector& outputIndices,
              const OutOfSampleWeightVector<BitWeightVector>& weights) const override final {
                return std::make_unique<
                  StatisticsSubset<StatisticVector, typename StatisticMatrix::view_type, RuleEvaluationFactory,
                                   OutOfSampleWeightVector<BitWeightVector>, PartialIndexVector>>(
                  statisticMatrixPtr_->getView(), *ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& outputIndices,
              const OutOfSampleWeightVector<DenseWeightVector<uint32>>& weights) const override final {
                return std::make_unique<
                  StatisticsSubset<StatisticVector, typename StatisticMatrix::view_type, RuleEvaluationFactory,
                                   OutOfSampleWeightVector<DenseWeightVector<uint32>>, CompleteIndexVector>>(
                  statisticMatrixPtr_->getView(), *ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const PartialIndexVector& outputIndices,
              const OutOfSampleWeightVector<DenseWeightVector<uint32>>& weights) const override final {
                return std::make_unique<
                  StatisticsSubset<StatisticVector, typename StatisticMatrix::view_type, RuleEvaluationFactory,
                                   OutOfSampleWeightVector<DenseWeightVector<uint32>>, PartialIndexVector>>(
                  statisticMatrixPtr_->getView(), *ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createWeightedStatistics`
             */
            std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
              const EqualWeightVector& weights) const override final {
                return std::make_unique<WeightedStatistics<StatisticVector, typename StatisticMatrix::view_type,
                                                           RuleEvaluationFactory, EqualWeightVector>>(
                  statisticMatrixPtr_->getView(), *ruleEvaluationFactory_, weights);
            }

            /**
             * @see `IStatistics::createWeightedStatistics`
             */
            std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
              const BitWeightVector& weights) const override final {
                return std::make_unique<WeightedStatistics<StatisticVector, typename StatisticMatrix::view_type,
                                                           RuleEvaluationFactory, BitWeightVector>>(
                  statisticMatrixPtr_->getView(), *ruleEvaluationFactory_, weights);
            }

            /**
             * @see `IStatistics::createWeightedStatistics`
             */
            std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
              const DenseWeightVector<uint32>& weights) const override final {
                return std::make_unique<WeightedStatistics<StatisticVector, typename StatisticMatrix::view_type,
                                                           RuleEvaluationFactory, DenseWeightVector<uint32>>>(
                  statisticMatrixPtr_->getView(), *ruleEvaluationFactory_, weights);
            }
    };

}
