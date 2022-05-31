/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/statistics/statistics_example_wise.hpp"


namespace boosting {

    template<typename WeightVector, typename StatisticView, typename StatisticVector>
    static inline void addExampleWiseStatistic(const WeightVector& weights, const StatisticView& statisticView,
                                               StatisticVector& statisticVector, uint32 statisticIndex) {
        float64 weight = weights.getWeight(statisticIndex);
        statisticVector.add(statisticView.gradients_row_cbegin(statisticIndex),
                            statisticView.gradients_row_cend(statisticIndex),
                            statisticView.hessians_row_cbegin(statisticIndex),
                            statisticView.hessians_row_cend(statisticIndex), weight);
    }

    template<typename StatisticView, typename StatisticVector>
    static inline void addExampleWiseStatistic(const EqualWeightVector& weights, const StatisticView& statisticView,
                                               StatisticVector& statisticVector, uint32 statisticIndex) {
        statisticVector.add(statisticView.gradients_row_cbegin(statisticIndex),
                            statisticView.gradients_row_cend(statisticIndex),
                            statisticView.hessians_row_cbegin(statisticIndex),
                            statisticView.hessians_row_cend(statisticIndex));
    }

    template<typename WeightVector, typename StatisticView, typename StatisticVector>
    static inline void removeExampleWiseStatistic(const WeightVector& weights, const StatisticView& statisticView,
                                                  StatisticVector& statisticVector, uint32 statisticIndex) {
        float64 weight = weights.getWeight(statisticIndex);
        statisticVector.remove(statisticView.gradients_row_cbegin(statisticIndex),
                               statisticView.gradients_row_cend(statisticIndex),
                               statisticView.hessians_row_cbegin(statisticIndex),
                               statisticView.hessians_row_cend(statisticIndex), weight);
    }

    template<typename StatisticView, typename StatisticVector>
    static inline void removeExampleWiseStatistic(const EqualWeightVector& weights, const StatisticView& statisticView,
                                                  StatisticVector& statisticVector, uint32 statisticIndex) {
        statisticVector.remove(statisticView.gradients_row_cbegin(statisticIndex),
                               statisticView.gradients_row_cend(statisticIndex),
                               statisticView.hessians_row_cbegin(statisticIndex),
                               statisticView.hessians_row_cend(statisticIndex));
    }

    template<typename Prediction, typename ScoreMatrix>
    static inline void applyExampleWisePredictionInternally(uint32 statisticIndex, const Prediction& prediction,
                                                            ScoreMatrix& scoreMatrix) {
        scoreMatrix.addToRowFromSubset(statisticIndex, prediction.scores_cbegin(), prediction.scores_cend(),
                                       prediction.indices_cbegin(), prediction.indices_cend());
    }

    template<typename LabelMatrix, typename StatisticView, typename ScoreMatrix, typename LossFunction>
    static inline void updateExampleWiseStatisticsInternally(uint32 statisticIndex, const LabelMatrix& labelMatrix,
                                                             StatisticView& statisticView, ScoreMatrix& scoreMatrix,
                                                             const LossFunction& lossFunction) {
        lossFunction.updateExampleWiseStatistics(statisticIndex, labelMatrix, scoreMatrix, statisticView);
    }

    /**
     * An abstract base class for all statistics that provide access to gradients and Hessians that are calculated
     * according to a differentiable loss function that is applied example-wise.
     *
     * @tparam StatisticVector          The type of the vectors that are used to store gradients and Hessians
     * @tparam StatisticView            The type of the view that provides access to the gradients and Hessians
     * @tparam RuleEvaluationFactory    The type of the factory that allows to create instances of the class that is
     *                                  used for calculating the predictions of rules, as well as corresponding quality
     *                                  scores
     */
    template<typename StatisticVector, typename StatisticView, typename RuleEvaluationFactory>
    class AbstractExampleWiseImmutableWeightedStatistics : virtual public IImmutableWeightedStatistics {

        protected:

            /**
             * An abstract base class for all subsets of the gradients and Hessians that are stored by an instance of
             * the class `AbstractExampleWiseImmutableWeightedStatistics`.
             *
             * @tparam IndexVector The type of the vector that provides access to the indices of the labels that are
             *                     included in the subset
             */
            template<typename IndexVector>
            class AbstractWeightedStatisticsSubset : public IWeightedStatisticsSubset {

                private:

                    const IndexVector& labelIndices_;

                    StatisticVector sumVector_;

                    StatisticVector tmpVector_;

                    std::unique_ptr<StatisticVector> accumulatedSumVectorPtr_;

                    std::unique_ptr<IRuleEvaluation<StatisticVector>> ruleEvaluationPtr_;

                protected:

                    /**
                     * A reference to an object of template type `StatisticView` that provides access to the gradients
                     * and Hessians.
                     */
                    const StatisticView& statisticView_;

                    /**
                     * A pointer to an object of template type `StatisticVector` that stores the total sum of all
                     * gradients and Hessians.
                     */
                    const StatisticVector* totalSumVector_;

                public:

                    /**
                     * @param statistics            A reference to an object of type
                     *                              `AbstractExampleWiseImmutableWeightedStatistics` that stores the
                     *                              gradients and Hessians
                     * @param totalSumVector        A reference to an object of template type `StatisticVector` that
                     *                              stores the total sums of gradients and Hessians
                     * @param ruleEvaluationFactory A reference to an object of template type `RuleEvaluationFactory`
                     *                              that allows to create instances of the class that should be used for
                     *                              calculating the predictions of rules, as well as corresponding
                     *                              quality scores
                     * @param labelIndices          A reference to an object of template type `IndexVector` that
                     *                              provides access to the indices of the labels that are included in
                     *                              the subset
                     */
                    AbstractWeightedStatisticsSubset(const AbstractExampleWiseImmutableWeightedStatistics& statistics,
                                                     const StatisticVector& totalSumVector,
                                                     const RuleEvaluationFactory& ruleEvaluationFactory,
                                                     const IndexVector& labelIndices)
                        : labelIndices_(labelIndices), sumVector_(StatisticVector(labelIndices.getNumElements(), true)),
                          tmpVector_(StatisticVector(labelIndices.getNumElements())),
                          ruleEvaluationPtr_(ruleEvaluationFactory.create(sumVector_, labelIndices)),
                          statisticView_(statistics.statisticView_), totalSumVector_(&totalSumVector) {

                    }

                    /**
                     * @see `IStatisticsSubset::addToSubset`
                     */
                    void addToSubset(uint32 statisticIndex, float64 weight) override final {
                        sumVector_.addToSubset(statisticView_.gradients_row_cbegin(statisticIndex),
                                               statisticView_.gradients_row_cend(statisticIndex),
                                               statisticView_.hessians_row_cbegin(statisticIndex),
                                               statisticView_.hessians_row_cend(statisticIndex), labelIndices_, weight);
                    }

                    /**
                     * @see `IWeightedStatisticsSubset::resetSubset`
                     */
                    void resetSubset() override final {
                        if (!accumulatedSumVectorPtr_) {
                            // Create a vector for storing the accumulated sums of gradients and Hessians, if
                            // necessary...
                            accumulatedSumVectorPtr_ = std::make_unique<StatisticVector>(sumVector_);
                        } else {
                            // Add the sum of gradients and Hessians to the accumulated sums of gradients and
                            // Hessians...
                            accumulatedSumVectorPtr_->add(sumVector_.gradients_cbegin(), sumVector_.gradients_cend(),
                                                          sumVector_.hessians_cbegin(), sumVector_.hessians_cend());
                        }

                        // Reset the sum of gradients and Hessians to zero...
                        sumVector_.clear();
                    }

                    /**
                     * @see `IStatisticsSubset::evaluate`
                     */
                    const IScoreVector& evaluate() override final {
                        return ruleEvaluationPtr_->evaluate(sumVector_);
                    }

                    /**
                     * @see `IWeightedStatisticsSubset::evaluateAccumulated`
                     */
                    const IScoreVector& evaluateAccumulated() override final {
                        return ruleEvaluationPtr_->evaluate(*accumulatedSumVectorPtr_);
                    }

                    /**
                     * @see `IWeightedStatisticsSubset::evaluateUncovered`
                     */
                    const IScoreVector& evaluateUncovered() override final {
                        tmpVector_.difference(totalSumVector_->gradients_cbegin(), totalSumVector_->gradients_cend(),
                                              totalSumVector_->hessians_cbegin(), totalSumVector_->hessians_cend(),
                                              labelIndices_, sumVector_.gradients_cbegin(), sumVector_.gradients_cend(),
                                              sumVector_.hessians_cbegin(), sumVector_.hessians_cend());
                        return ruleEvaluationPtr_->evaluate(tmpVector_);
                    }

                    /**
                     * @see `IWeightedStatisticsSubset::evaluateUncoveredAccumulated`
                     */
                    const IScoreVector& evaluateUncoveredAccumulated() override final {
                        tmpVector_.difference(totalSumVector_->gradients_cbegin(), totalSumVector_->gradients_cend(),
                                              totalSumVector_->hessians_cbegin(), totalSumVector_->hessians_cend(),
                                              labelIndices_, accumulatedSumVectorPtr_->gradients_cbegin(),
                                              accumulatedSumVectorPtr_->gradients_cend(),
                                              accumulatedSumVectorPtr_->hessians_cbegin(),
                                              accumulatedSumVectorPtr_->hessians_cend());
                        return ruleEvaluationPtr_->evaluate(tmpVector_);
                    }

            };

        protected:

            /**
             * A reference to an object of template type `StatisticView` that stores the gradients and Hessians.
             */
            const StatisticView& statisticView_;

            /**
             * A reference to an object of template type `RuleEvaluationFactory` that allows to create instances of the
             * class that is used for calculating the predictions of rules, as well as corresponding quality scores.
             */
            const RuleEvaluationFactory& ruleEvaluationFactory_;

        public:

            /**
             * @param statisticView         A reference to an object of template type `StatisticView` that provides
             *                              access to the gradients and Hessians
             * @param ruleEvaluationFactory A reference to an object of template type `RuleEvaluationFactory` that
             *                              allows to create instances of the class that should be used for calculating
             *                              the predictions of rules, as well as corresponding quality scores
             */
            AbstractExampleWiseImmutableWeightedStatistics(const StatisticView& statisticView,
                                                           const RuleEvaluationFactory& ruleEvaluationFactory)
                : statisticView_(statisticView), ruleEvaluationFactory_(ruleEvaluationFactory) {

            }

            /**
             * @see `IImmutableWeightedStatistics::getNumStatistics`
             */
            uint32 getNumStatistics() const override final {
                return statisticView_.getNumRows();
            }

            /**
             * @see `IImmutableWeightedStatistics::getNumLabels`
             */
            uint32 getNumLabels() const override final {
                return statisticView_.getNumCols();
            }

    };

    /**
     * Provides access to gradients and Hessians that are calculated according to a differentiable loss function that is
     * applied example-wise and are organized as a histogram.
     *
     * @tparam StatisticVector          The type of the vectors that are used to store gradients and Hessians
     * @tparam StatisticView            The type of the view that provides access to the original gradients and Hessians
     * @tparam Histogram                The type of a histogram that stores aggregated gradients and Hessians
     * @tparam RuleEvaluationFactory    The type of the factory that allows to create instances of the class that is
     *                                  used for calculating the predictions of rules, as well as corresponding quality
     *                                  scores
     */
    template<typename StatisticVector, typename StatisticView, typename Histogram, typename RuleEvaluationFactory>
    class ExampleWiseHistogram final : public AbstractExampleWiseImmutableWeightedStatistics<StatisticVector,
                                                                                             Histogram,
                                                                                             RuleEvaluationFactory>,
                                       virtual public IHistogram {

        private:

            /**
             * Provides access to a subset of the gradients and Hessians that are stored by an instance of the class
             * `ExampleWiseHistogram`.
             *
             * @tparam IndexVector The type of the vector that provides access to the indices of the labels that are
             *                     included in the subset
             */
            template<typename IndexVector>
            class WeightedStatisticsSubset final :
                    public AbstractExampleWiseImmutableWeightedStatistics<StatisticVector, Histogram,
                                                                          RuleEvaluationFactory>::template AbstractWeightedStatisticsSubset<IndexVector> {

                private:

                    const ExampleWiseHistogram& histogram_;

                    std::unique_ptr<StatisticVector> totalCoverableSumVectorPtr_;

                public:

                    /**
                     * @param histogram             A reference to an object of type `ExampleWiseHistogram` that stores
                     *                              the gradients and Hessians
                     * @param totalSumVector        A reference to an object of template type `StatisticVector` that
                     *                              stores the total sums of gradients and Hessians
                     * @param ruleEvaluationFactory A reference to an object of template type `RuleEvaluationFactory`
                     *                              that allows to create instances of the class that should be used for
                     *                              calculating the predictions of rules, as well as corresponding
                     *                              quality scores
                     * @param labelIndices          A reference to an object of template type `IndexVector` that
                     *                              provides access to the indices of the labels that are included in
                     *                              the subset
                     */
                    WeightedStatisticsSubset(const ExampleWiseHistogram& histogram,
                                             const StatisticVector& totalSumVector,
                                             const RuleEvaluationFactory& ruleEvaluationFactory,
                                             const IndexVector& labelIndices)
                        : AbstractExampleWiseImmutableWeightedStatistics<StatisticVector, Histogram,
                                                                         RuleEvaluationFactory>::template AbstractWeightedStatisticsSubset<IndexVector>(
                              histogram, totalSumVector, ruleEvaluationFactory, labelIndices),
                          histogram_(histogram) {

                    }

                    /**
                     * @see `IWeightedStatisticsSubset::addToMissing`
                     */
                    void addToMissing(uint32 statisticIndex, float64 weight) override {
                        // Create a vector for storing the totals sums of gradients and Hessians, if necessary...
                        if (!totalCoverableSumVectorPtr_) {
                            totalCoverableSumVectorPtr_ = std::make_unique<StatisticVector>(*this->totalSumVector_);
                            this->totalSumVector_ = totalCoverableSumVectorPtr_.get();
                        }

                        // Subtract the gradients and Hessians of the example at the given index (weighted by the given
                        // weight) from the total sums of gradients and Hessians...
                        const StatisticView& originalStatisticView = histogram_.originalStatisticView_;
                        totalCoverableSumVectorPtr_->remove(originalStatisticView.gradients_row_cbegin(statisticIndex),
                                                            originalStatisticView.gradients_row_cend(statisticIndex),
                                                            originalStatisticView.hessians_row_cbegin(statisticIndex),
                                                            originalStatisticView.hessians_row_cend(statisticIndex),
                                                            weight);
                    }

            };

            std::unique_ptr<Histogram> histogramPtr_;

            const StatisticView& originalStatisticView_;

            const StatisticVector& totalSumVector_;

        public:

            /**
             * @param histogramPtr          An unique pointer to an object of template type `Histogram` that stores the
             *                              gradients and Hessians in the histogram
             * @param originalStatisticView A reference to an object of template type `StatisticView` that provides
             *                              access to the original gradients and Hessians, the histogram was created
             *                              from
             * @param totalSumVector        A reference to an object of template type `StatisticVector` that stores the
             *                              total sums of gradients and Hessians
             * @param ruleEvaluationFactory A reference to an object of type `RuleEvaluationFactory` that allows to
             *                              create instances of the class that should be used for calculating the
             *                              predictions of rules, as well as corresponding quality scores
             */
            ExampleWiseHistogram(std::unique_ptr<Histogram> histogramPtr, const StatisticView& originalStatisticView,
                                 const StatisticVector& totalSumVector,
                                 const RuleEvaluationFactory& ruleEvaluationFactory)
                : AbstractExampleWiseImmutableWeightedStatistics<StatisticVector, Histogram, RuleEvaluationFactory>(
                      *histogramPtr, ruleEvaluationFactory),
                  histogramPtr_(std::move(histogramPtr)), originalStatisticView_(originalStatisticView),
                  totalSumVector_(totalSumVector) {

            }

            /**
             * @see `IHistogram::clear`
             */
            void clear() override {
                histogramPtr_->clear();
            }

            /**
             * @see `IHistogram::addToBin`
             */
            void addToBin(uint32 binIndex, uint32 statisticIndex, float64 weight) override {
                histogramPtr_->addToRow(binIndex, originalStatisticView_.gradients_row_cbegin(statisticIndex),
                                        originalStatisticView_.gradients_row_cend(statisticIndex),
                                        originalStatisticView_.hessians_row_cbegin(statisticIndex),
                                        originalStatisticView_.hessians_row_cend(statisticIndex), weight);
            }

            /**
             * @see `IImmutableWeightedStatistics::createSubset`
             */
            std::unique_ptr<IWeightedStatisticsSubset> createSubset(
                    const CompleteIndexVector& labelIndices) const override final {
                return std::make_unique<WeightedStatisticsSubset<CompleteIndexVector>>(*this, totalSumVector_,
                                                                                       this->ruleEvaluationFactory_,
                                                                                       labelIndices);
            }

            /**
             * @see `IImmutableWeightedStatistics::createSubset`
             */
            std::unique_ptr<IWeightedStatisticsSubset> createSubset(
                    const PartialIndexVector& labelIndices) const override final {
                return std::make_unique<WeightedStatisticsSubset<PartialIndexVector>>(*this, totalSumVector_,
                                                                                      this->ruleEvaluationFactory_,
                                                                                      labelIndices);
            }

    };

    /**
     * Provides access to weighted gradients and Hessians that are calculated according to a differentiable loss
     * function that is applied example-wise and allows to update the gradients and Hessians after a new rule has been
     * learned.
     *
     * @tparam WeightVector             The type of the vector that provides access to the weights of individual
     *                                  statistics
     * @tparam StatisticVector          The type of the vectors that are used to store gradients and Hessians
     * @tparam StatisticView            The type of the view that provides access to the gradients and Hessians
     * @tparam Histogram                The type of a histogram that stores aggregated gradients and Hessians
     * @tparam RuleEvaluationFactory    The type of the factory that allows to create instances of the class that is
     *                                  used for calculating the predictions of rules, as well as corresponding quality
     *                                  scores
     */
    template<typename WeightVector, typename StatisticVector, typename StatisticView, typename Histogram,
             typename RuleEvaluationFactory>
    class ExampleWiseWeightedStatistics :
            public AbstractExampleWiseImmutableWeightedStatistics<StatisticVector, StatisticView,
                                                                  RuleEvaluationFactory>,
            virtual public IWeightedStatistics {

        private:


            /**
             * Provides access to a subset of the gradients and Hessians that are stored by an instance of the class
             * `ExampleWiseWeightedStatistics`.
             *
             * @tparam IndexVector The type of the vector that provides access to the indices of the labels that are
             *                     included in the subset
             */
            template<typename IndexVector>
            class WeightedStatisticsSubset final :
                    public AbstractExampleWiseImmutableWeightedStatistics<StatisticVector, StatisticView,
                                                                          RuleEvaluationFactory>::template AbstractWeightedStatisticsSubset<IndexVector> {

                private:

                    std::unique_ptr<StatisticVector> totalCoverableSumVectorPtr_;

                public:

                    /**
                     * @param statistics            A reference to an object of type `ExampleWiseWeightedStatistics`
                     *                              that stores the gradients and Hessians
                     * @param totalSumVector        A reference to an object of template type `StatisticVector` that
                     *                              stores the total sums of gradients and Hessians
                     * @param ruleEvaluationFactory A reference to an object of template type `RuleEvaluationFactory`
                     *                              that allows to create instances of the class that should be used for
                     *                              calculating the predictions of rules, as well as corresponding
                     *                              quality scores
                     * @param labelIndices          A reference to an object of template type `IndexVector` that
                     *                              provides access to the indices of the labels that are included in
                     *                              the subset
                     */
                    WeightedStatisticsSubset(const ExampleWiseWeightedStatistics& statistics,
                                             const StatisticVector& totalSumVector,
                                             const RuleEvaluationFactory& ruleEvaluationFactory,
                                             const IndexVector& labelIndices)
                        : AbstractExampleWiseImmutableWeightedStatistics<StatisticVector, StatisticView,
                                                                         RuleEvaluationFactory>::template AbstractWeightedStatisticsSubset<IndexVector>(
                              statistics, totalSumVector, ruleEvaluationFactory, labelIndices) {

                    }

                    /**
                     * @see `IWeightedStatisticsSubset::addToMissing`
                     */
                    void addToMissing(uint32 statisticIndex, float64 weight) override {
                        // Create a vector for storing the totals sums of gradients and Hessians, if necessary...
                        if (!totalCoverableSumVectorPtr_) {
                            totalCoverableSumVectorPtr_ = std::make_unique<StatisticVector>(*this->totalSumVector_);
                            this->totalSumVector_ = totalCoverableSumVectorPtr_.get();
                        }

                        // Subtract the gradients and Hessians of the example at the given index (weighted by the given
                        // weight) from the total sums of gradients and Hessians...
                        totalCoverableSumVectorPtr_->remove(this->statisticView_.gradients_row_cbegin(statisticIndex),
                                                            this->statisticView_.gradients_row_cend(statisticIndex),
                                                            this->statisticView_.hessians_row_cbegin(statisticIndex),
                                                            this->statisticView_.hessians_row_cend(statisticIndex),
                                                            weight);
                    }

            };

            const WeightVector& weights_;

            std::unique_ptr<StatisticVector> totalSumVectorPtr_;

        public:

            /**
             * @param weights               A reference to an object of template type `WeightVector` that provides
             *                              access to the weights of individual statistics
             * @param statisticView         A reference to an object of template type `StatisticView` that provides
             *                              access to the gradients and Hessians
             * @param ruleEvaluationFactory A reference to an object of template type `RuleEvaluationFactory` that
             *                              allows to create instances of the class that should be used for calculating
             *                              the predictions of rules, as well as corresponding quality scores
             */
            ExampleWiseWeightedStatistics(const WeightVector& weights, const StatisticView& statisticView,
                                          const RuleEvaluationFactory& ruleEvaluationFactory)
                : AbstractExampleWiseImmutableWeightedStatistics<StatisticVector, StatisticView, RuleEvaluationFactory>(
                      statisticView, ruleEvaluationFactory),
                  weights_(weights),
                  totalSumVectorPtr_(std::make_unique<StatisticVector>(statisticView.getNumCols(), true)) {
                uint32 numStatistics = weights.getNumElements();

                for (uint32 i = 0; i < numStatistics; i++) {
                    addExampleWiseStatistic(weights, statisticView, *totalSumVectorPtr_, i);
                }
            }

            /**
             * @see `IWeightedStatistics::resetCoveredStatistics`
             */
            void resetCoveredStatistics() override final {
                totalSumVectorPtr_->clear();
            }

            /**
             * @see `IWeightedStatistics::addCoveredStatistic`
             */
            void addCoveredStatistic(uint32 statisticIndex) override final {
                addExampleWiseStatistic(weights_, this->statisticView_, *totalSumVectorPtr_, statisticIndex);
            }

            /**
             * @see `IWeightedStatistics::removeCoveredStatistic`
             */
            void removeCoveredStatistic(uint32 statisticIndex) override final {
                removeExampleWiseStatistic(weights_, this->statisticView_, *totalSumVectorPtr_, statisticIndex);
            }

            /**
             * @see `IWeightedStatistics::createHistogram`
             */
            std::unique_ptr<IHistogram> createHistogram(uint32 numBins) const override final {
                const StatisticView& originalStatisticView = this->statisticView_;
                std::unique_ptr<Histogram> histogramPtr =
                    std::make_unique<Histogram>(numBins, originalStatisticView.getNumCols());
                return std::make_unique<ExampleWiseHistogram<StatisticVector, StatisticView, Histogram,
                                                             RuleEvaluationFactory>>(std::move(histogramPtr),
                                                                                     originalStatisticView,
                                                                                     *totalSumVectorPtr_,
                                                                                     this->ruleEvaluationFactory_);
            }

            /**
             * @see `IImmutableWeightedStatistics::createSubset`
             */
            std::unique_ptr<IWeightedStatisticsSubset> createSubset(
                    const CompleteIndexVector& labelIndices) const override final {
                return std::make_unique<WeightedStatisticsSubset<CompleteIndexVector>>(*this, *totalSumVectorPtr_,
                                                                                       this->ruleEvaluationFactory_,
                                                                                       labelIndices);
            }

            /**
             * @see `IImmutableWeightedStatistics::createSubset`
             */
            std::unique_ptr<IWeightedStatisticsSubset> createSubset(
                    const PartialIndexVector& labelIndices) const override final {
                return std::make_unique<WeightedStatisticsSubset<PartialIndexVector>>(*this, *totalSumVectorPtr_,
                                                                                      this->ruleEvaluationFactory_,
                                                                                      labelIndices);
            }

    };

    /**
     * An abstract base class for all statistics that provide access to gradients and Hessians that are calculated
     * according to a differentiable loss function that is applied example-wise.
     *
     * @tparam LabelMatrix                      The type of the matrix that provides access to the labels of the
     *                                          training examples
     * @tparam StatisticVector                  The type of the vectors that are used to store gradients and Hessians
     * @tparam StatisticView                    The type of the view that provides access to the gradients and Hessians
     * @tparam Histogram                        The type of a histogram that stores aggregated gradients and Hessians
     * @tparam ScoreMatrix                      The type of the matrices that are used to store predicted scores
     * @tparam LossFunction                     The type of the loss function that is used to calculate gradients and
     *                                          Hessians
     * @tparam EvaluationMeasure                The type of the evaluation measure that is used to assess the quality of
     *                                          predictions for a specific statistic
     * @tparam ExampleWiseRuleEvaluationFactory The type of the factory that allows to create instances of the class
     *                                          that is used for calculating the example-wise predictions of rules , as
     *                                          well as corresponding quality scores
     * @tparam LabelWiseRuleEvaluationFactory   The type of the factory that allows to create instances of the class
     *                                          that is used for calculating the label-wise predictions of rules, as
     *                                          well as corresponding quality scores
     */
    template<typename LabelMatrix, typename StatisticVector, typename StatisticView, typename Histogram,
             typename ScoreMatrix, typename LossFunction, typename EvaluationMeasure,
             typename ExampleWiseRuleEvaluationFactory, typename LabelWiseRuleEvaluationFactory>
    class AbstractExampleWiseStatistics :
            virtual public IExampleWiseStatistics<ExampleWiseRuleEvaluationFactory, LabelWiseRuleEvaluationFactory> {

        private:

            const ExampleWiseRuleEvaluationFactory* ruleEvaluationFactory_;

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
             * The label matrix that provides access to the labels of the training examples.
             */
            const LabelMatrix& labelMatrix_;

            /**
             * An unique pointer to an object of template type `StatisticView` that stores the gradients and Hessians.
             */
            std::unique_ptr<StatisticView> statisticViewPtr_;

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
             * @param ruleEvaluationFactory A reference to an object of template type `ExampleWiseRuleEvaluationFactory`
             *                              that allows to create instances of the class that should be used for
             *                              calculating the predictions of rules, as well as corresponding quality
             *                              scores
             * @param labelMatrix           A reference to an object of template type `LabelMatrix` that provides access
             *                              to the labels of the training examples
             * @param statisticViewPtr      An unique pointer to an object of template type `StatisticView` that
             *                              provides access to the gradients and Hessians
             * @param scoreMatrixPtr        An unique pointer to an object of template type `ScoreMatrix` that stores
             *                              the currently predicted scores
             */
            AbstractExampleWiseStatistics(std::unique_ptr<LossFunction> lossPtr,
                                          std::unique_ptr<EvaluationMeasure> evaluationMeasurePtr,
                                          const ExampleWiseRuleEvaluationFactory& ruleEvaluationFactory,
                                          const LabelMatrix& labelMatrix,
                                          std::unique_ptr<StatisticView> statisticViewPtr,
                                          std::unique_ptr<ScoreMatrix> scoreMatrixPtr)
                : ruleEvaluationFactory_(&ruleEvaluationFactory), lossPtr_(std::move(lossPtr)),
                  evaluationMeasurePtr_(std::move(evaluationMeasurePtr)), labelMatrix_(labelMatrix),
                  statisticViewPtr_(std::move(statisticViewPtr)), scoreMatrixPtr_(std::move(scoreMatrixPtr)) {

            }

            /**
             * @see `IExampleWiseStatistics::setRuleEvaluationFactory`
             */
            void setRuleEvaluationFactory(
                    const ExampleWiseRuleEvaluationFactory& ruleEvaluationFactory) override final {
                this->ruleEvaluationFactory_ = &ruleEvaluationFactory;
            }

            /**
             * @see `IStatistics::getNumStatistics`
             */
            uint32 getNumStatistics() const override final {
                return statisticViewPtr_->getNumRows();
            }

            /**
             * @see `IStatistics::getNumLabels`
             */
            uint32 getNumLabels() const override final {
                return statisticViewPtr_->getNumCols();
            }

            /**
             * @see `IStatistics::applyPrediction`
             */
            void applyPrediction(uint32 statisticIndex, const CompletePrediction& prediction) override final {
                applyExampleWisePredictionInternally(statisticIndex, prediction, *scoreMatrixPtr_);
                updateExampleWiseStatisticsInternally(statisticIndex, labelMatrix_, *this->statisticViewPtr_,
                                                      *scoreMatrixPtr_, *lossPtr_);
            }

            /**
             * @see `IStatistics::applyPrediction`
             */
            void applyPrediction(uint32 statisticIndex, const PartialPrediction& prediction) override final {
                applyExampleWisePredictionInternally(statisticIndex, prediction, *scoreMatrixPtr_);
                updateExampleWiseStatisticsInternally(statisticIndex, labelMatrix_, *this->statisticViewPtr_,
                                                      *scoreMatrixPtr_, *lossPtr_);
            }

            /**
             * @see `IStatistics::evaluatePrediction`
             */
            float64 evaluatePrediction(uint32 statisticIndex) const override final {
                return evaluationMeasurePtr_->evaluate(statisticIndex, labelMatrix_, *scoreMatrixPtr_);
            }

            /**
             * @see `IStatistics::createWeightedStatistics`
             */
            std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
                    const EqualWeightVector& weights) const override final {
                return std::make_unique<ExampleWiseWeightedStatistics<EqualWeightVector, StatisticVector, StatisticView,
                                                                      Histogram, ExampleWiseRuleEvaluationFactory>>(
                    weights, *statisticViewPtr_, *ruleEvaluationFactory_);
            }

            /**
             * @see `IStatistics::createWeightedStatistics`
             */
            std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
                    const BitWeightVector& weights) const override final {
                return std::make_unique<ExampleWiseWeightedStatistics<BitWeightVector, StatisticVector, StatisticView,
                                                                      Histogram, ExampleWiseRuleEvaluationFactory>>(
                    weights, *statisticViewPtr_, *ruleEvaluationFactory_);
            }

            /**
             * @see `IStatistics::createWeightedStatistics`
             */
            std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
                    const DenseWeightVector<uint32>& weights) const override final {
                return std::make_unique<ExampleWiseWeightedStatistics<DenseWeightVector<uint32>, StatisticVector,
                                                                      StatisticView, Histogram,
                                                                      ExampleWiseRuleEvaluationFactory>>(
                    weights, *statisticViewPtr_, *ruleEvaluationFactory_);
            }

    };

}
