/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/statistics/statistics_non_decomposable.hpp"

#include <memory>
#include <utility>

#include <memory>
#include <utility>

namespace boosting {

    static inline bool hasNonZeroWeightNonDecomposable(const EqualWeightVector& weights, uint32 statisticIndex) {
        return true;
    }

    template<typename WeightVector>
    static inline bool hasNonZeroWeightNonDecomposable(const WeightVector& weights, uint32 statisticIndex) {
        return !isEqualToZero(weights[statisticIndex]);
    }

    template<typename StatisticView, typename StatisticVector, typename IndexVector>
    static inline void addNonDecomposableStatisticToSubset(const EqualWeightVector& weights,
                                                           const StatisticView& statisticView, StatisticVector& vector,
                                                           const IndexVector& outputIndices, uint32 statisticIndex) {
        vector.addToSubset(statisticView.gradients_cbegin(statisticIndex), statisticView.gradients_cend(statisticIndex),
                           statisticView.hessians_cbegin(statisticIndex), statisticView.hessians_cend(statisticIndex),
                           outputIndices);
    }

    template<typename WeightVector, typename StatisticView, typename StatisticVector, typename IndexVector>
    static inline void addNonDecomposableStatisticToSubset(const WeightVector& weights,
                                                           const StatisticView& statisticView, StatisticVector& vector,
                                                           const IndexVector& outputIndices, uint32 statisticIndex) {
        float64 weight = weights[statisticIndex];
        vector.addToSubset(statisticView.gradients_cbegin(statisticIndex), statisticView.gradients_cend(statisticIndex),
                           statisticView.hessians_cbegin(statisticIndex), statisticView.hessians_cend(statisticIndex),
                           outputIndices, weight);
    }

    /**
     * A subset of gradients and Hessians that are calculated according to a non-decomposable loss function and are
     * accessible via a view.
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
    class NonDecomposableStatisticsSubset : virtual public IStatisticsSubset {
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
            NonDecomposableStatisticsSubset(const StatisticView& statisticView,
                                            const RuleEvaluationFactory& ruleEvaluationFactory,
                                            const WeightVector& weights, const IndexVector& outputIndices)
                : sumVector_(outputIndices.getNumElements(), true), statisticView_(statisticView), weights_(weights),
                  outputIndices_(outputIndices),
                  ruleEvaluationPtr_(ruleEvaluationFactory.create(sumVector_, outputIndices)) {}

            /**
             * @see `IStatisticsSubset::hasNonZeroWeight`
             */
            bool hasNonZeroWeight(uint32 statisticIndex) const override final {
                return hasNonZeroWeightNonDecomposable(weights_, statisticIndex);
            }

            /**
             * @see `IStatisticsSubset::addToSubset`
             */
            void addToSubset(uint32 statisticIndex) override final {
                addNonDecomposableStatisticToSubset(weights_, statisticView_, sumVector_, outputIndices_,
                                                    statisticIndex);
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
     * according to a non-decomposable loss function.
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
    class AbstractNonDecomposableImmutableWeightedStatistics : virtual public IImmutableWeightedStatistics {
        protected:

            /**
             * An abstract base class for all subsets of the gradients and Hessians that are stored by an instance of
             * the class `AbstractNonDecomposableImmutableWeightedStatistics`.
             *
             * @tparam IndexVector The type of the vector that provides access to the indices of the outputs that are
             *                     included in the subset
             */
            template<typename IndexVector>
            class AbstractWeightedStatisticsSubset
                : public NonDecomposableStatisticsSubset<StatisticVector, StatisticView, RuleEvaluationFactory,
                                                         WeightVector, IndexVector>,
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
                     * @param statistics        A reference to an object of type
                     *                          `AbstractNonDecomposableImmutableWeightedStatistics` that stores the
                     *                          gradients and Hessians
                     * @param totalSumVector    A reference to an object of template type `StatisticVector` that stores
                     *                          the total sums of gradients and Hessians
                     * @param outputIndices     A reference to an object of template type `IndexVector` that provides
                     *                          access to the indices of the outputs that are included in the subset
                     */
                    AbstractWeightedStatisticsSubset(
                      const AbstractNonDecomposableImmutableWeightedStatistics& statistics,
                      const StatisticVector& totalSumVector, const IndexVector& outputIndices)
                        : NonDecomposableStatisticsSubset<StatisticVector, StatisticView, RuleEvaluationFactory,
                                                          WeightVector, IndexVector>(
                            statistics.statisticView_, statistics.ruleEvaluationFactory_, statistics.weights_,
                            outputIndices),
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
                            accumulatedSumVectorPtr_->add(
                              this->sumVector_.gradients_cbegin(), this->sumVector_.gradients_cend(),
                              this->sumVector_.hessians_cbegin(), this->sumVector_.hessians_cend());
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
                        tmpVector_.difference(totalSumVector_->gradients_cbegin(), totalSumVector_->gradients_cend(),
                                              totalSumVector_->hessians_cbegin(), totalSumVector_->hessians_cend(),
                                              this->outputIndices_, this->sumVector_.gradients_cbegin(),
                                              this->sumVector_.gradients_cend(), this->sumVector_.hessians_cbegin(),
                                              this->sumVector_.hessians_cend());
                        return this->ruleEvaluationPtr_->calculateScores(tmpVector_);
                    }

                    /**
                     * @see `IWeightedStatisticsSubset::calculateScoresUncoveredAccumulated`
                     */
                    const IScoreVector& calculateScoresUncoveredAccumulated() override final {
                        tmpVector_.difference(
                          totalSumVector_->gradients_cbegin(), totalSumVector_->gradients_cend(),
                          totalSumVector_->hessians_cbegin(), totalSumVector_->hessians_cend(), this->outputIndices_,
                          accumulatedSumVectorPtr_->gradients_cbegin(), accumulatedSumVectorPtr_->gradients_cend(),
                          accumulatedSumVectorPtr_->hessians_cbegin(), accumulatedSumVectorPtr_->hessians_cend());
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
            AbstractNonDecomposableImmutableWeightedStatistics(const StatisticView& statisticView,
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

    template<typename WeightVector, typename StatisticView, typename StatisticVector>
    static inline void addNonDecomposableStatistic(const WeightVector& weights, const StatisticView& statisticView,
                                                   StatisticVector& statisticVector, uint32 statisticIndex) {
        float64 weight = weights[statisticIndex];
        statisticVector.add(statisticView.gradients_cbegin(statisticIndex),
                            statisticView.gradients_cend(statisticIndex), statisticView.hessians_cbegin(statisticIndex),
                            statisticView.hessians_cend(statisticIndex), weight);
    }

    template<typename StatisticView, typename StatisticVector>
    static inline void addNonDecomposableStatistic(const EqualWeightVector& weights, const StatisticView& statisticView,
                                                   StatisticVector& statisticVector, uint32 statisticIndex) {
        statisticVector.add(statisticView.gradients_cbegin(statisticIndex),
                            statisticView.gradients_cend(statisticIndex), statisticView.hessians_cbegin(statisticIndex),
                            statisticView.hessians_cend(statisticIndex));
    }

    template<typename WeightVector, typename StatisticView, typename StatisticVector>
    static inline void removeNonDecomposableStatistic(const WeightVector& weights, const StatisticView& statisticView,
                                                      StatisticVector& statisticVector, uint32 statisticIndex) {
        float64 weight = weights[statisticIndex];
        statisticVector.remove(
          statisticView.gradients_cbegin(statisticIndex), statisticView.gradients_cend(statisticIndex),
          statisticView.hessians_cbegin(statisticIndex), statisticView.hessians_cend(statisticIndex), weight);
    }

    template<typename StatisticView, typename StatisticVector>
    static inline void removeNonDecomposableStatistic(const EqualWeightVector& weights,
                                                      const StatisticView& statisticView,
                                                      StatisticVector& statisticVector, uint32 statisticIndex) {
        statisticVector.remove(
          statisticView.gradients_cbegin(statisticIndex), statisticView.gradients_cend(statisticIndex),
          statisticView.hessians_cbegin(statisticIndex), statisticView.hessians_cend(statisticIndex));
    }

    /**
     * Provides access to weighted gradients and Hessians that are calculated according to a non-decomposable loss
     * function and allows to update the gradients and Hessians after a new rule has been learned.
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
    class NonDecomposableWeightedStatistics final
        : virtual public IWeightedStatistics,
          public AbstractNonDecomposableImmutableWeightedStatistics<StatisticVector, StatisticView,
                                                                    RuleEvaluationFactory, WeightVector> {
        private:

            /**
             * Provides access to a subset of the gradients and Hessians that are stored by an instance of the class
             * `NonDecomposableWeightedStatistics`.
             *
             * @tparam IndexVector The type of the vector that provides access to the indices of the outputs that are
             *                     included in the subset
             */
            template<typename IndexVector>
            class WeightedStatisticsSubset final
                : public AbstractNonDecomposableImmutableWeightedStatistics<
                    StatisticVector, StatisticView, RuleEvaluationFactory,
                    WeightVector>::template AbstractWeightedStatisticsSubset<IndexVector> {
                private:

                    std::unique_ptr<StatisticVector> totalCoverableSumVectorPtr_;

                public:

                    /**
                     * @param statistics        A reference to an object of type `NonDecomposableWeightedStatistics`
                     *                          that stores the gradients and Hessians
                     * @param totalSumVector    A reference to an object of template type `StatisticVector` that stores
                     *                          the total sums of gradients and Hessians
                     * @param outputIndices     A reference to an object of template type `IndexVector` that provides
                     *                          access to the indices of the outputs that are included in the subset
                     */
                    WeightedStatisticsSubset(const NonDecomposableWeightedStatistics& statistics,
                                             const StatisticVector& totalSumVector, const IndexVector& outputIndices)
                        : AbstractNonDecomposableImmutableWeightedStatistics<
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
                        removeNonDecomposableStatistic(this->weights_, this->statisticView_,
                                                       *totalCoverableSumVectorPtr_, statisticIndex);
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
            NonDecomposableWeightedStatistics(const StatisticView& statisticView,
                                              const RuleEvaluationFactory& ruleEvaluationFactory,
                                              const WeightVector& weights)
                : AbstractNonDecomposableImmutableWeightedStatistics<StatisticVector, StatisticView,
                                                                     RuleEvaluationFactory, WeightVector>(
                    statisticView, ruleEvaluationFactory, weights),
                  totalSumVectorPtr_(std::make_unique<StatisticVector>(statisticView.numCols, true)) {
                uint32 numStatistics = weights.getNumElements();

                for (uint32 i = 0; i < numStatistics; i++) {
                    addNonDecomposableStatistic(weights, statisticView, *totalSumVectorPtr_, i);
                }
            }

            /**
             * @param statistics A reference to an object of type `NonDecomposableWeightedStatistics` to be copied
             */
            NonDecomposableWeightedStatistics(const NonDecomposableWeightedStatistics& statistics)
                : AbstractNonDecomposableImmutableWeightedStatistics<StatisticVector, StatisticView,
                                                                     RuleEvaluationFactory, WeightVector>(
                    statistics.statisticView_, statistics.ruleEvaluationFactory_, statistics.weights_),
                  totalSumVectorPtr_(std::make_unique<StatisticVector>(*statistics.totalSumVectorPtr_)) {}

            /**
             * @see `IWeightedStatistics::copy`
             */
            std::unique_ptr<IWeightedStatistics> copy() const override {
                return std::make_unique<NonDecomposableWeightedStatistics<StatisticVector, StatisticView,
                                                                          RuleEvaluationFactory, WeightVector>>(*this);
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
                addNonDecomposableStatistic(this->weights_, this->statisticView_, *totalSumVectorPtr_, statisticIndex);
            }

            /**
             * @see `IWeightedStatistics::removeCoveredStatistic`
             */
            void removeCoveredStatistic(uint32 statisticIndex) override {
                removeNonDecomposableStatistic(this->weights_, this->statisticView_, *totalSumVectorPtr_,
                                               statisticIndex);
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

    template<typename OutputMatrix, typename StatisticView, typename ScoreMatrix, typename LossFunction>
    static inline void updateNonDecomposableStatisticsInternally(uint32 statisticIndex,
                                                                 const OutputMatrix& outputMatrix,
                                                                 StatisticView& statisticView, ScoreMatrix& scoreMatrix,
                                                                 const LossFunction& lossFunction) {
        lossFunction.updateNonDecomposableStatistics(statisticIndex, outputMatrix, scoreMatrix.getView(),
                                                     statisticView);
    }

    /**
     * An abstract base class for all statistics that provide access to gradients and Hessians that are calculated
     * according to a non-decomposable loss function.
     *
     * @tparam OutputMatrix                         The type of the matrix that provides access to the ground truth of
     *                                              the training examples
     * @tparam StatisticVector                      The type of the vectors that are used to store gradients and
     *                                              Hessians
     * @tparam StatisticMatrix                      The type of the matrix that stores the gradients and Hessians
     * @tparam ScoreMatrix                          The type of the matrices that are used to store predicted scores
     * @tparam LossFunction                         The type of the loss function that is used to calculate gradients
     *                                              and Hessians
     * @tparam EvaluationMeasure                    The type of the evaluation measure that is used to assess the
     *                                              quality of predictions for a specific statistic
     * @tparam NonDecomposableRuleEvaluationFactory The type of the factory that allows to create instances of the class
     *                                              that is used for calculating the predictions of rules, as well as
     *                                              their overall quality, based on gradients and Hessians that have
     *                                              been calculated according to a non-decomposable loss function
     * @tparam DecomposableRuleEvaluationFactory    The type of the factory that allows to create instances of the class
     *                                              that is used for calculating the predictions of rules, as well as
     *                                              their overall quality, based on gradients and Hessians that have
     *                                              been calculated according to a decomposable loss function
     */
    template<typename OutputMatrix, typename StatisticVector, typename StatisticMatrix, typename ScoreMatrix,
             typename LossFunction, typename EvaluationMeasure, typename NonDecomposableRuleEvaluationFactory,
             typename DecomposableRuleEvaluationFactory>
    class AbstractNonDecomposableStatistics
        : virtual public INonDecomposableStatistics<NonDecomposableRuleEvaluationFactory,
                                                    DecomposableRuleEvaluationFactory> {
        private:

            const NonDecomposableRuleEvaluationFactory* ruleEvaluationFactory_;

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
             * The output matrix that provides access to the ground truth of the training examples.
             */
            const OutputMatrix& outputMatrix_;

            /**
             * An unique pointer to an object of template type `StatisticMatrix` that stores the gradients and Hessians.
             */
            const std::unique_ptr<StatisticMatrix> statisticMatrixPtr_;

            /**
             * The score matrix that stores the currently predicted scores.
             */
            std::unique_ptr<ScoreMatrix> scoreMatrixPtr_;

        public:

            /**
             * @param lossPtr               An unique pointer to an object of template type `LossFunction` that
             *                              implements the loss function that should be used for calculating gradients
             *                              and Hessians
             * @param evaluationMeasurePtr  An unique pointer to an object of template type `EvaluationMeasure` that
             *                              implements the evaluation measure that should be used to assess the quality
             *                              of predictions for a specific statistic
             * @param ruleEvaluationFactory A reference to an object of template type
             *                              `NonDecomposableRuleEvaluationFactory` that allows to create instances of
             *                              the class that should be used for calculating the predictions of rules, as
             *                              well as their overall quality
             * @param outputMatrix          A reference to an object of template type `OutputMatrix` that provides
             *                              access to the ground truth of the training examples
             * @param statisticMatrixPtr    An unique pointer to an object of template type `StatisticView` that stores
             *                              the gradients and Hessians
             * @param scoreMatrixPtr        An unique pointer to an object of template type `ScoreMatrix` that stores
             *                              the currently predicted scores
             */
            AbstractNonDecomposableStatistics(std::unique_ptr<LossFunction> lossPtr,
                                              std::unique_ptr<EvaluationMeasure> evaluationMeasurePtr,
                                              const NonDecomposableRuleEvaluationFactory& ruleEvaluationFactory,
                                              const OutputMatrix& outputMatrix,
                                              std::unique_ptr<StatisticMatrix> statisticMatrixPtr,
                                              std::unique_ptr<ScoreMatrix> scoreMatrixPtr)
                : ruleEvaluationFactory_(&ruleEvaluationFactory), lossPtr_(std::move(lossPtr)),
                  evaluationMeasurePtr_(std::move(evaluationMeasurePtr)), outputMatrix_(outputMatrix),
                  statisticMatrixPtr_(std::move(statisticMatrixPtr)), scoreMatrixPtr_(std::move(scoreMatrixPtr)) {}

            /**
             * @see `INonDecomposableStatistics::setRuleEvaluationFactory`
             */
            void setRuleEvaluationFactory(
              const NonDecomposableRuleEvaluationFactory& ruleEvaluationFactory) override final {
                this->ruleEvaluationFactory_ = &ruleEvaluationFactory;
            }

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
                updateNonDecomposableStatisticsInternally(statisticIndex, outputMatrix_, statisticMatrixPtr_->getView(),
                                                          *scoreMatrixPtr_, *lossPtr_);
            }

            /**
             * @see `IStatistics::applyPrediction`
             */
            void applyPrediction(uint32 statisticIndex, const PartialPrediction& prediction) override final {
                applyPredictionInternally(statisticIndex, prediction, *scoreMatrixPtr_);
                updateNonDecomposableStatisticsInternally(statisticIndex, outputMatrix_, statisticMatrixPtr_->getView(),
                                                          *scoreMatrixPtr_, *lossPtr_);
            }

            /**
             * @see `IStatistics::revertPrediction`
             */
            void revertPrediction(uint32 statisticIndex, const CompletePrediction& prediction) override final {
                revertPredictionInternally(statisticIndex, prediction, *scoreMatrixPtr_);
                updateNonDecomposableStatisticsInternally(statisticIndex, outputMatrix_, statisticMatrixPtr_->getView(),
                                                          *scoreMatrixPtr_, *lossPtr_);
            }

            /**
             * @see `IStatistics::revertPrediction`
             */
            void revertPrediction(uint32 statisticIndex, const PartialPrediction& prediction) override final {
                revertPredictionInternally(statisticIndex, prediction, *scoreMatrixPtr_);
                updateNonDecomposableStatisticsInternally(statisticIndex, outputMatrix_, statisticMatrixPtr_->getView(),
                                                          *scoreMatrixPtr_, *lossPtr_);
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
                return std::make_unique<NonDecomposableStatisticsSubset<
                  StatisticVector, typename StatisticMatrix::view_type, NonDecomposableRuleEvaluationFactory,
                  EqualWeightVector, CompleteIndexVector>>(statisticMatrixPtr_->getView(), *ruleEvaluationFactory_,
                                                           weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& outputIndices,
                                                            const EqualWeightVector& weights) const override final {
                return std::make_unique<NonDecomposableStatisticsSubset<
                  StatisticVector, typename StatisticMatrix::view_type, NonDecomposableRuleEvaluationFactory,
                  EqualWeightVector, PartialIndexVector>>(statisticMatrixPtr_->getView(), *ruleEvaluationFactory_,
                                                          weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const CompleteIndexVector& outputIndices,
                                                            const BitWeightVector& weights) const override final {
                return std::make_unique<NonDecomposableStatisticsSubset<
                  StatisticVector, typename StatisticMatrix::view_type, NonDecomposableRuleEvaluationFactory,
                  BitWeightVector, CompleteIndexVector>>(statisticMatrixPtr_->getView(), *ruleEvaluationFactory_,
                                                         weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& outputIndices,
                                                            const BitWeightVector& weights) const override final {
                return std::make_unique<NonDecomposableStatisticsSubset<
                  StatisticVector, typename StatisticMatrix::view_type, NonDecomposableRuleEvaluationFactory,
                  BitWeightVector, PartialIndexVector>>(statisticMatrixPtr_->getView(), *ruleEvaluationFactory_,
                                                        weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& outputIndices, const DenseWeightVector<uint32>& weights) const override final {
                return std::make_unique<NonDecomposableStatisticsSubset<
                  StatisticVector, typename StatisticMatrix::view_type, NonDecomposableRuleEvaluationFactory,
                  DenseWeightVector<uint32>, CompleteIndexVector>>(statisticMatrixPtr_->getView(),
                                                                   *ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const PartialIndexVector& outputIndices, const DenseWeightVector<uint32>& weights) const override final {
                return std::make_unique<NonDecomposableStatisticsSubset<
                  StatisticVector, typename StatisticMatrix::view_type, NonDecomposableRuleEvaluationFactory,
                  DenseWeightVector<uint32>, PartialIndexVector>>(statisticMatrixPtr_->getView(),
                                                                  *ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& outputIndices,
              const OutOfSampleWeightVector<EqualWeightVector>& weights) const override final {
                return std::make_unique<NonDecomposableStatisticsSubset<
                  StatisticVector, typename StatisticMatrix::view_type, NonDecomposableRuleEvaluationFactory,
                  OutOfSampleWeightVector<EqualWeightVector>, CompleteIndexVector>>(
                  statisticMatrixPtr_->getView(), *ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const PartialIndexVector& outputIndices,
              const OutOfSampleWeightVector<EqualWeightVector>& weights) const override final {
                return std::make_unique<NonDecomposableStatisticsSubset<
                  StatisticVector, typename StatisticMatrix::view_type, NonDecomposableRuleEvaluationFactory,
                  OutOfSampleWeightVector<EqualWeightVector>, PartialIndexVector>>(
                  statisticMatrixPtr_->getView(), *ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& outputIndices,
              const OutOfSampleWeightVector<BitWeightVector>& weights) const override final {
                return std::make_unique<NonDecomposableStatisticsSubset<
                  StatisticVector, typename StatisticMatrix::view_type, NonDecomposableRuleEvaluationFactory,
                  OutOfSampleWeightVector<BitWeightVector>, CompleteIndexVector>>(
                  statisticMatrixPtr_->getView(), *ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const PartialIndexVector& outputIndices,
              const OutOfSampleWeightVector<BitWeightVector>& weights) const override final {
                return std::make_unique<NonDecomposableStatisticsSubset<
                  StatisticVector, typename StatisticMatrix::view_type, NonDecomposableRuleEvaluationFactory,
                  OutOfSampleWeightVector<BitWeightVector>, PartialIndexVector>>(
                  statisticMatrixPtr_->getView(), *ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& outputIndices,
              const OutOfSampleWeightVector<DenseWeightVector<uint32>>& weights) const override final {
                return std::make_unique<NonDecomposableStatisticsSubset<
                  StatisticVector, typename StatisticMatrix::view_type, NonDecomposableRuleEvaluationFactory,
                  OutOfSampleWeightVector<DenseWeightVector<uint32>>, CompleteIndexVector>>(
                  statisticMatrixPtr_->getView(), *ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const PartialIndexVector& outputIndices,
              const OutOfSampleWeightVector<DenseWeightVector<uint32>>& weights) const override final {
                return std::make_unique<NonDecomposableStatisticsSubset<
                  StatisticVector, typename StatisticMatrix::view_type, NonDecomposableRuleEvaluationFactory,
                  OutOfSampleWeightVector<DenseWeightVector<uint32>>, PartialIndexVector>>(
                  statisticMatrixPtr_->getView(), *ruleEvaluationFactory_, weights, outputIndices);
            }

            /**
             * @see `IStatistics::createWeightedStatistics`
             */
            std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
              const EqualWeightVector& weights) const override final {
                return std::make_unique<
                  NonDecomposableWeightedStatistics<StatisticVector, typename StatisticMatrix::view_type,
                                                    NonDecomposableRuleEvaluationFactory, EqualWeightVector>>(
                  statisticMatrixPtr_->getView(), *ruleEvaluationFactory_, weights);
            }

            /**
             * @see `IStatistics::createWeightedStatistics`
             */
            std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
              const BitWeightVector& weights) const override final {
                return std::make_unique<
                  NonDecomposableWeightedStatistics<StatisticVector, typename StatisticMatrix::view_type,
                                                    NonDecomposableRuleEvaluationFactory, BitWeightVector>>(
                  statisticMatrixPtr_->getView(), *ruleEvaluationFactory_, weights);
            }

            /**
             * @see `IStatistics::createWeightedStatistics`
             */
            std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
              const DenseWeightVector<uint32>& weights) const override final {
                return std::make_unique<
                  NonDecomposableWeightedStatistics<StatisticVector, typename StatisticMatrix::view_type,
                                                    NonDecomposableRuleEvaluationFactory, DenseWeightVector<uint32>>>(
                  statisticMatrixPtr_->getView(), *ruleEvaluationFactory_, weights);
            }
    };

}
