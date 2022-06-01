/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/statistics/statistics_label_wise.hpp"


namespace boosting {

    template<typename StatisticView, typename StatisticVector>
    static inline void addLabelWiseStatistic(const EqualWeightVector& weights, const StatisticView& statisticView,
                                             StatisticVector& statisticVector, uint32 statisticIndex) {
        statisticVector.add(statisticView, statisticIndex);
    }

    template<typename WeightVector, typename StatisticView, typename StatisticVector>
    static inline void addLabelWiseStatistic(const WeightVector& weights, const StatisticView& statisticView,
                                             StatisticVector& statisticVector, uint32 statisticIndex) {
        float64 weight = weights.getWeight(statisticIndex);
        statisticVector.add(statisticView, statisticIndex, weight);
    }

    template<typename StatisticView, typename StatisticVector>
    static inline void removeLabelWiseStatistic(const EqualWeightVector& weights, const StatisticView& statisticView,
                                                StatisticVector& statisticVector, uint32 statisticIndex) {
        statisticVector.remove(statisticView, statisticIndex);
    }

    template<typename WeightVector, typename StatisticView, typename StatisticVector>
    static inline void removeLabelWiseStatistic(const WeightVector& weights, const StatisticView& statisticView,
                                                StatisticVector& statisticVector, uint32 statisticIndex) {
        float64 weight = weights.getWeight(statisticIndex);
        statisticVector.remove(statisticView, statisticIndex, weight);
    }

    template<typename Prediction, typename ScoreMatrix>
    static inline void applyLabelWisePredictionInternally(uint32 statisticIndex, const Prediction& prediction,
                                                          ScoreMatrix& scoreMatrix) {
        scoreMatrix.addToRowFromSubset(statisticIndex, prediction.scores_cbegin(), prediction.scores_cend(),
                                       prediction.indices_cbegin(), prediction.indices_cend());
    }

    template<typename Prediction, typename LabelMatrix, typename StatisticView, typename ScoreMatrix,
             typename LossFunction>
    static inline void updateLabelWiseStatisticsInternally(uint32 statisticIndex, const Prediction& prediction,
                                                           const LabelMatrix& labelMatrix, StatisticView& statisticView,
                                                           ScoreMatrix& scoreMatrix, const LossFunction& lossFunction) {
        lossFunction.updateLabelWiseStatistics(statisticIndex, labelMatrix, scoreMatrix, prediction.indices_cbegin(),
                                               prediction.indices_cend(), statisticView);
    }

    /**
     * A subset of gradients and Hessians that are calculated according to a differentiable loss function that is
     * applied label-wise and are accessible via a view.
     *
     * @tparam StatisticVector          The type of the vector that is used to store the sums of gradients and Hessians
     * @tparam StatisticView            The type of the view that provides access to the gradients and Hessians
     * @tparam RuleEvaluationFactory    The type of the factory that allows to create instances of the class that is
     *                                  used for calculating the predictions of rules, as well as corresponding quality
     *                                  scores
     * @tparam IndexVector              The type of the vector that provides access to the indices of the labels that
     *                                  are included in the subset
     */
    template<typename StatisticVector, typename StatisticView, typename RuleEvaluationFactory, typename IndexVector>
    class LabelWiseStatisticsSubset : virtual public IStatisticsSubset {

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
             * A reference to an object of template type `IndexVector` that provides access to the indices of the labels
             * that are included in the subset.
             */
            const IndexVector& labelIndices_;

            /**
             * An unique pointer to an object of type `IRuleEvaluation` that is used to calculate the predictions of
             * rules, as well as corresponding quality scores.
             */
            std::unique_ptr<IRuleEvaluation<StatisticVector>> ruleEvaluationPtr_;

        public:

            /**
             * @param statisticView         A reference to an object of template type `StatisticView` that provides
             *                              access to the gradients and Hessians
             * @param ruleEvaluationFactory A reference to an object of template type `RuleEvaluationFactory` that
             *                              allows to create instances of the class that is used for calculating the
             *                              predictions of rules, as well as corresponding quality scores
             * @param labelIndices          A reference to an object of template type `IndexVector` that provides access
             *                              to the indices of the labels that are included in the subset
             */
            LabelWiseStatisticsSubset(const StatisticView& statisticView,
                                      const RuleEvaluationFactory& ruleEvaluationFactory,
                                      const IndexVector& labelIndices)
                : sumVector_(StatisticVector(labelIndices.getNumElements(), true)), statisticView_(statisticView),
                  labelIndices_(labelIndices),
                  ruleEvaluationPtr_(ruleEvaluationFactory.create(sumVector_, labelIndices)) {

            }

            /**
             * @see `IStatisticsSubset::addToSubset`
             */
            void addToSubset(uint32 statisticIndex, float64 weight) override final {
                sumVector_.addToSubset(statisticView_, statisticIndex, labelIndices_, weight);
            }

            /**
             * @see `IStatisticsSubset::evaluate`
             */
            const IScoreVector& evaluate() override final {
                return ruleEvaluationPtr_->evaluate(sumVector_);
            }

    };

    /**
     * An abstract base class for all statistics that provide access to gradients and Hessians that are calculated
     * according to a differentiable loss function that is applied label-wise.
     *
     * @tparam StatisticVector          The type of the vectors that are used to store gradients and Hessians
     * @tparam StatisticView            The type of the view that provides access to the gradients and Hessians
     * @tparam RuleEvaluationFactory    The type of the factory that allows to create instances of the class that is
     *                                  used for calculating the predictions of rules, as well as corresponding quality
     *                                  scores
     */
    template<typename StatisticVector, typename StatisticView, typename RuleEvaluationFactory>
    class AbstractLabelWiseImmutableWeightedStatistics : virtual public IImmutableWeightedStatistics {

        protected:

            /**
             * An abstract base class for all subsets of the gradients and Hessians that are stored by an instance of
             * the class `AbstractLabelWiseImmutableWeightedStatistics`.
             *
             * @tparam IndexVector The type of the vector that provides access to the indices of the labels that are
             *                     included in the subset
             */
            template<typename IndexVector>
            class AbstractWeightedStatisticsSubset : public LabelWiseStatisticsSubset<StatisticVector, StatisticView,
                                                                                      RuleEvaluationFactory,
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
                     * @param statistics            A reference to an object of type
                     *                              `AbstractLabelWiseImmutableWeightedStatistics` that stores the
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
                    AbstractWeightedStatisticsSubset(const AbstractLabelWiseImmutableWeightedStatistics& statistics,
                                                     const StatisticVector& totalSumVector,
                                                     const RuleEvaluationFactory& ruleEvaluationFactory,
                                                     const IndexVector& labelIndices)
                        : LabelWiseStatisticsSubset<StatisticVector, StatisticView, RuleEvaluationFactory, IndexVector>(
                              statistics.statisticView_, ruleEvaluationFactory, labelIndices),
                          tmpVector_(StatisticVector(labelIndices.getNumElements())), totalSumVector_(&totalSumVector) {

                    }

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
                     * @see `IWeightedStatisticsSubset::evaluateAccumulated`
                     */
                    const IScoreVector& evaluateAccumulated() override final {
                        return this->ruleEvaluationPtr_->evaluate(*accumulatedSumVectorPtr_);
                    }

                    /**
                     * @see `IWeightedStatisticsSubset::evaluateUncovered`
                     */
                    const IScoreVector& evaluateUncovered() override final {
                        tmpVector_.difference(*totalSumVector_, this->labelIndices_, this->sumVector_);
                        return this->ruleEvaluationPtr_->evaluate(tmpVector_);
                    }

                    /**
                     * @see `IWeightedStatisticsSubset::evaluateUncoveredAccumulated`
                     */
                    const IScoreVector& evaluateUncoveredAccumulated() override final {
                        tmpVector_.difference(*totalSumVector_, this->labelIndices_, *accumulatedSumVectorPtr_);
                        return this->ruleEvaluationPtr_->evaluate(tmpVector_);
                    }

            };

        protected:

            /**
             * A reference to an object of template type `StatisticView` that stores the gradients and Hessians.
             */
            const StatisticView& statisticView_;

            /**
             * A reference to an object of template type `RuleEvaluationFactory` that is used to create instances of the
             * class that is used for calculating the predictions of rules, as well as corresponding quality scores.
             */
            const RuleEvaluationFactory& ruleEvaluationFactory_;

        public:

            /**
             * @param statisticView         A reference to an object of template type `StatisticView` that provides
             *                              access to the the gradients and Hessians
             * @param ruleEvaluationFactory A reference to an object of template type `RuleEvaluationFactory`, that
             *                              allows to create instances of the class that should be used for calculating
             *                              the predictions of rules, as well as corresponding quality scores
             */
            AbstractLabelWiseImmutableWeightedStatistics(const StatisticView& statisticView,
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
     * applied label-wise and are organized as a histogram.
     *
     * @tparam StatisticVector          The type of the vectors that are used to store gradients and Hessians
     * @tparam StatisticView            The type of the view that provides access to the original gradients and Hessians
     * @tparam Histogram                The type of a histogram that stores aggregated gradients and Hessians
     * @tparam RuleEvaluationFactory    The type of the factory that allows to create instances of the class that is
     *                                  used for calculating the predictions of rules, as well as corresponding quality
     *                                  scores
     */
    template<typename StatisticVector, typename StatisticView, typename Histogram, typename RuleEvaluationFactory>
    class LabelWiseHistogram final : virtual public IHistogram,
                                     public AbstractLabelWiseImmutableWeightedStatistics<StatisticVector, Histogram,
                                                                                         RuleEvaluationFactory> {

        private:

            /**
             * Provides access to a subset of the gradients and Hessians that are stored by an instance of the class
             * `LabelWiseHistogram`.
             *
             * @tparam IndexVector The type of the vector that provides access to the indices of the labels that are
             *                     included in the subset
             */
            template<typename IndexVector>
            class WeightedStatisticsSubset final :
                    public AbstractLabelWiseImmutableWeightedStatistics<StatisticVector, Histogram,
                                                                        RuleEvaluationFactory>::template AbstractWeightedStatisticsSubset<IndexVector> {

                private:

                    const LabelWiseHistogram& histogram_;

                    std::unique_ptr<StatisticVector> totalCoverableSumVectorPtr_;

                public:

                    /**
                     * @param histogram             A reference to an object of type `LabelWiseHistogram` that stores
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
                    WeightedStatisticsSubset(const LabelWiseHistogram& histogram, const StatisticVector& totalSumVector,
                                             const RuleEvaluationFactory& ruleEvaluationFactory,
                                             const IndexVector& labelIndices)
                        : AbstractLabelWiseImmutableWeightedStatistics<StatisticVector, Histogram,
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
                        totalCoverableSumVectorPtr_->remove(originalStatisticView, statisticIndex, weight);
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
            LabelWiseHistogram(std::unique_ptr<Histogram> histogramPtr, const StatisticView& originalStatisticView,
                               const StatisticVector& totalSumVector,
                               const RuleEvaluationFactory& ruleEvaluationFactory)
                : AbstractLabelWiseImmutableWeightedStatistics<StatisticVector, Histogram, RuleEvaluationFactory>(
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
                histogramPtr_->addToRow(binIndex, originalStatisticView_.row_cbegin(statisticIndex),
                                        originalStatisticView_.row_cend(statisticIndex), weight);
            }

            /**
             * @see `IImmutableWeightedStatistics::createSubset`
             */
            std::unique_ptr<IWeightedStatisticsSubset> createSubset(
                    const CompleteIndexVector& labelIndices) const override {
                return std::make_unique<WeightedStatisticsSubset<CompleteIndexVector>>(*this, totalSumVector_,
                                                                                       this->ruleEvaluationFactory_,
                                                                                       labelIndices);
            }

            /**
             * @see `IImmutableWeightedStatistics::createSubset`
             */
            std::unique_ptr<IWeightedStatisticsSubset> createSubset(
                    const PartialIndexVector& labelIndices) const override {
                return std::make_unique<WeightedStatisticsSubset<PartialIndexVector>>(*this, totalSumVector_,
                                                                                      this->ruleEvaluationFactory_,
                                                                                      labelIndices);
            }

    };

    /**
     * An abstract base class for all classes that provide access to weighted gradients and Hessians that are calculated
     * according to a differentiable loss function that is applied label-wise and allows to update the gradients and
     * Hessians after a new rule has been learned.
     *
     * @tparam StatisticVector          The type of the vectors that are used to store gradients and Hessians
     * @tparam StatisticView            The type of the view that provides access to the gradients and Hessians
     * @tparam Histogram                The type of a histogram that stores aggregated gradients and Hessians
     * @tparam RuleEvaluationFactory    The type of the factory that allows to create instances of the class that is
     *                                  used for calculating the predictions of rules, as well as corresponding quality
     *                                  scores
     * @tparam WeightVector             The type of the vector that provides access to the weights of individual
     *                                  statistics
     */
    template<typename StatisticVector, typename StatisticView, typename Histogram, typename RuleEvaluationFactory,
             typename WeightVector>
    class LabelWiseWeightedStatistics : virtual public IWeightedStatistics,
                                        public AbstractLabelWiseImmutableWeightedStatistics<StatisticVector,
                                                                                            StatisticView,
                                                                                            RuleEvaluationFactory> {

        private:

            /**
             * Provides access to a subset of the gradients and Hessians that are stored by an instance of the class
             * `LabelWiseWeightedStatistics`.
             *
             * @tparam IndexVector The type of the vector that provides access to the indices of the labels that are
             *                     included in the subset
             */
            template<typename IndexVector>
            class WeightedStatisticsSubset final :
                    public AbstractLabelWiseImmutableWeightedStatistics<StatisticVector, StatisticView,
                                                                        RuleEvaluationFactory>::template AbstractWeightedStatisticsSubset<IndexVector> {

                private:

                    std::unique_ptr<StatisticVector> totalCoverableSumVectorPtr_;

                public:

                    /**
                     * @param statistics            A reference to an object of type `LabelWiseWeightedStatistics` that
                     *                              stores the gradients and Hessians
                     * @param totalSumVector        A reference to an object of template type `StatisticVector` that
                     *                              stores the total sums of gradients and Hessians
                     * @param ruleEvaluationFactory A reference to an object of type `RuleEvaluationFactory` that allows
                     *                              to create instances of the class that should be used for calculating
                     *                              the predictions of rules, as well as corresponding quality scores
                     * @param labelIndices          A reference to an object of template type `IndexVector` that
                     *                              provides access to the indices of the labels that are included in
                     *                              the subset
                     */
                    WeightedStatisticsSubset(const LabelWiseWeightedStatistics& statistics,
                                             const StatisticVector& totalSumVector,
                                             const RuleEvaluationFactory& ruleEvaluationFactory,
                                             const IndexVector& labelIndices)
                        : AbstractLabelWiseImmutableWeightedStatistics<StatisticVector, StatisticView,
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
                        totalCoverableSumVectorPtr_->remove(this->statisticView_, statisticIndex, weight);
                    }

            };

            const WeightVector& weights_;

            std::unique_ptr<StatisticVector> totalSumVectorPtr_;

        public:

            /**
             * @param statisticView         A reference to an object of template type `StatisticView` that provides
             *                              access to the gradients and Hessians
             * @param ruleEvaluationFactory A reference to an object of type `RuleEvaluationFactory` that allows to
             *                              create instances of the class that should be used for calculating the
             *                              predictions of rules, as well as corresponding quality scores
             * @param weights               A reference to an object of template type `WeightVector` that provides
             *                              access to the weights of individual statistics
             */
            LabelWiseWeightedStatistics(const StatisticView& statisticView,
                                        const RuleEvaluationFactory& ruleEvaluationFactory,
                                        const WeightVector& weights)
                : AbstractLabelWiseImmutableWeightedStatistics<StatisticVector, StatisticView, RuleEvaluationFactory>(
                      statisticView, ruleEvaluationFactory),
                  weights_(weights),
                  totalSumVectorPtr_(std::make_unique<StatisticVector>(statisticView.getNumCols(), true)) {
                uint32 numStatistics = weights.getNumElements();

                for (uint32 i = 0; i < numStatistics; i++) {
                    addLabelWiseStatistic(weights, statisticView, *totalSumVectorPtr_, i);
                }
            }

            /**
             * @see `IWeightedStatistics::resetCoveredStatistic`
             */
            void resetCoveredStatistics() override final {
                totalSumVectorPtr_->clear();
            }

            /**
             * @see `IWeightedStatistics::addCoveredStatistic`
             */
            void addCoveredStatistic(uint32 statisticIndex) override final {
                addLabelWiseStatistic(weights_, this->statisticView_, *totalSumVectorPtr_, statisticIndex);
            }

            /**
             * @see `IWeightedStatistics::removeCoveredStatistic`
             */
            void removeCoveredStatistic(uint32 statisticIndex) override final {
                removeLabelWiseStatistic(weights_, this->statisticView_, *totalSumVectorPtr_, statisticIndex);
            }

            /**
             * @see `IWeightedStatistics::createHistogram`
             */
            std::unique_ptr<IHistogram> createHistogram(uint32 numBins) const override final {
                const StatisticView& originalStatisticView = this->statisticView_;
                std::unique_ptr<Histogram> histogramPtr =
                    std::make_unique<Histogram>(numBins, originalStatisticView.getNumCols());
                return std::make_unique<LabelWiseHistogram<StatisticVector, StatisticView, Histogram,
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
     * according to a differentiable loss function that is applied label-wise.
     *
     * @tparam LabelMatrix              The type of the matrix that provides access to the labels of the training
     *                                  examples
     * @tparam StatisticVector          The type of the vectors that are used to store gradients and Hessians
     * @tparam StatisticView            The type of the view that provides access to the gradients and Hessians
     * @tparam Histogram                The type of a histogram that stores aggregated gradients and Hessians
     * @tparam ScoreMatrix              The type of the matrices that are used to store predicted scores
     * @tparam LossFunction             The type of the loss function that is used to calculate gradients and Hessians
     * @tparam EvaluationMeasure        The type of the evaluation measure that is used to assess the quality of
     *                                  predictions for a specific statistic
     * @tparam RuleEvaluationFactory    The type of the factory that allows to create instances of the class that is
     *                                  used for calculating the predictions of rules, as well as corresponding quality
     *                                  scores
     */
    template<typename LabelMatrix, typename StatisticVector, typename StatisticView, typename Histogram,
             typename ScoreMatrix, typename LossFunction, typename EvaluationMeasure, typename RuleEvaluationFactory>
    class AbstractLabelWiseStatistics : virtual public ILabelWiseStatistics<RuleEvaluationFactory> {

        private:

            const std::unique_ptr<LossFunction> lossPtr_;

            const std::unique_ptr<EvaluationMeasure> evaluationMeasurePtr_;

            const RuleEvaluationFactory* ruleEvaluationFactory_;

            const LabelMatrix& labelMatrix_;

            std::unique_ptr<StatisticView> statisticViewPtr_;

            std::unique_ptr<ScoreMatrix> scoreMatrixPtr_;

        public:

            /**
             * @param lossPtr               An unique pointer to an object of template type `LossFunction` that
             *                              implements the loss function that should be used for calculating gradients
             *                              and Hessians
             * @param evaluationMeasurePtr  An unique pointer to an object of template type `EvaluationMeasure` that
             *                              implements the evaluation measure that should be used to assess the quality
             *                              of predictions for a specific statistic
             * @param ruleEvaluationFactory A reference to an object of type `RuleEvaluationFactory` that allows to
             *                              create instances of the class that should be used for calculating the
             *                              predictions of rules, as well as corresponding quality scores
             * @param labelMatrix           A reference to an object of template type `LabelMatrix` that provides access
             *                              to the labels of the training examples
             * @param statisticViewPtr      An unique pointer to an object of template type `StatisticView` that
             *                              provides access to the gradients and Hessians
             * @param scoreMatrixPtr        An unique pointer to an object of template type `ScoreMatrix` that stores
             *                              the currently predicted scores
             */
            AbstractLabelWiseStatistics(std::unique_ptr<LossFunction> lossPtr,
                                        std::unique_ptr<EvaluationMeasure> evaluationMeasurePtr,
                                        const RuleEvaluationFactory& ruleEvaluationFactory,
                                        const LabelMatrix& labelMatrix, std::unique_ptr<StatisticView> statisticViewPtr,
                                        std::unique_ptr<ScoreMatrix> scoreMatrixPtr)
                : lossPtr_(std::move(lossPtr)), evaluationMeasurePtr_(std::move(evaluationMeasurePtr)),
                  ruleEvaluationFactory_(&ruleEvaluationFactory), labelMatrix_(labelMatrix),
                  statisticViewPtr_(std::move(statisticViewPtr)), scoreMatrixPtr_(std::move(scoreMatrixPtr)) {

            }

            /**
             * @see `ILabelWiseStatistics::setRuleEvaluationFactory`
             */
            void setRuleEvaluationFactory(const RuleEvaluationFactory& ruleEvaluationFactory) override final {
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
                applyLabelWisePredictionInternally(statisticIndex, prediction, *scoreMatrixPtr_);
                updateLabelWiseStatisticsInternally(statisticIndex, prediction, labelMatrix_, *this->statisticViewPtr_,
                                                    *scoreMatrixPtr_, *lossPtr_);
            }

            /**
             * @see `IStatistics::applyPrediction`
             */
            void applyPrediction(uint32 statisticIndex, const PartialPrediction& prediction) override final {
                applyLabelWisePredictionInternally(statisticIndex, prediction, *scoreMatrixPtr_);
                updateLabelWiseStatisticsInternally(statisticIndex, prediction, labelMatrix_, *this->statisticViewPtr_,
                                                    *scoreMatrixPtr_, *lossPtr_);
            }

            /**
             * @see `IStatistics::evaluatePrediction`
             */
            float64 evaluatePrediction(uint32 statisticIndex) const override final {
                return evaluationMeasurePtr_->evaluate(statisticIndex, labelMatrix_, *scoreMatrixPtr_);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
                    const CompleteIndexVector& labelIndices) const override final {
                return std::make_unique<LabelWiseStatisticsSubset<StatisticVector, StatisticView, RuleEvaluationFactory,
                                                                  CompleteIndexVector>>(*statisticViewPtr_,
                                                                                        *ruleEvaluationFactory_,
                                                                                        labelIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
                    const PartialIndexVector& labelIndices) const override final {
                return std::make_unique<LabelWiseStatisticsSubset<StatisticVector, StatisticView, RuleEvaluationFactory,
                                                                  PartialIndexVector>>(*statisticViewPtr_,
                                                                                       *ruleEvaluationFactory_,
                                                                                       labelIndices);
            }

            /**
             * @see `IStatistics::createWeightedStatistics`
             */
            std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
                    const EqualWeightVector& weights) const override final {
                return std::make_unique<LabelWiseWeightedStatistics<StatisticVector, StatisticView, Histogram,
                                                                    RuleEvaluationFactory, EqualWeightVector>>(
                    *statisticViewPtr_, *ruleEvaluationFactory_, weights);
            }

            /**
             * @see `IStatistics::createWeightedStatistics`
             */
            std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
                    const BitWeightVector& weights) const override final {
                return std::make_unique<LabelWiseWeightedStatistics<StatisticVector, StatisticView, Histogram,
                                                                    RuleEvaluationFactory, BitWeightVector>>(
                    *statisticViewPtr_, *ruleEvaluationFactory_, weights);
            }

            /**
             * @see `IStatistics::createWeightedStatistics`
             */
            std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
                    const DenseWeightVector<uint32>& weights) const override final {
                return std::make_unique<LabelWiseWeightedStatistics<StatisticVector, StatisticView, Histogram,
                                                                    RuleEvaluationFactory, DenseWeightVector<uint32>>>(
                    *statisticViewPtr_, *ruleEvaluationFactory_, weights);
            }

    };

}
