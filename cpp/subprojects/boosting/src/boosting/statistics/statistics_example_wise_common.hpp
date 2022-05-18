/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/statistics/statistics_example_wise.hpp"


namespace boosting {

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
     * @tparam RuleEvaluationFactory    The type of the classes that may be used for calculating the predictions, as
     *                                  well as corresponding quality scores, of rules
     */
    template<typename StatisticVector, typename StatisticView, typename RuleEvaluationFactory>
    class AbstractExampleWiseImmutableStatistics : virtual public IImmutableStatistics {

        protected:

            /**
             * An abstract base class for all subsets of the gradients and Hessians that are stored by an instance of
             * the class `AbstractExampleWiseImmutableStatistics`.
             *
             * @tparam T The type of the vector that provides access to the indices of the labels that are included in
             *           the subset
             */
            template<typename T>
            class AbstractStatisticsSubset : public IStatisticsSubset {

                private:

                    const AbstractExampleWiseImmutableStatistics& statistics_;

                    std::unique_ptr<IRuleEvaluation<StatisticVector>> ruleEvaluationPtr_;

                    const T& labelIndices_;

                    StatisticVector sumVector_;

                    StatisticVector tmpVector_;

                    std::unique_ptr<StatisticVector> accumulatedSumVectorPtr_;

                protected:

                    /**
                     * A pointer to an object of template type `StatisticVector` that stores the total sum of all
                     * gradients and Hessians.
                     */
                    const StatisticVector* totalSumVector_;

                    /**
                     * Returns the view that provides access to the gradients and Hessians.
                     *
                     * @return A reference to an object of template type `StatisticView` that provides access to the
                     *         gradient and Hessians
                     */
                    const StatisticView& getStatisticView() {
                        return *statistics_.statisticViewPtr_;
                    }

                public:

                    /**
                     * @param statistics        A reference to an object of type
                     *                          `AbstractExampleWiseImmutableStatistics` that stores the gradients and
                     *                          Hessians
                     * @param totalSumVector    A reference to an object of template type `StatisticVector` that stores
                     *                          the total sums of gradients and Hessians
                     * @param ruleEvaluationPtr An unique pointer to an object of type `IRuleEvaluation` that should be
                     *                          used to calculate the predictions, as well as corresponding quality
                     *                          scores, of rules
                     * @param labelIndices      A reference to an object of template type `T` that provides access to
                     *                          the indices of the labels that are included in the subset
                     */
                    AbstractStatisticsSubset(const AbstractExampleWiseImmutableStatistics& statistics,
                                             const StatisticVector& totalSumVector,
                                             std::unique_ptr<IRuleEvaluation<StatisticVector>> ruleEvaluationPtr,
                                             const T& labelIndices)
                        : statistics_(statistics), ruleEvaluationPtr_(std::move(ruleEvaluationPtr)),
                          labelIndices_(labelIndices), sumVector_(StatisticVector(labelIndices.getNumElements(), true)),
                          tmpVector_(StatisticVector(labelIndices.getNumElements())), totalSumVector_(&totalSumVector) {

                    }

                    /**
                     * @see `IStatisticsSubset::addToSubset`
                     */
                    void addToSubset(uint32 statisticIndex, float64 weight) override final {
                        sumVector_.addToSubset(statistics_.statisticViewPtr_->gradients_row_cbegin(statisticIndex),
                                               statistics_.statisticViewPtr_->gradients_row_cend(statisticIndex),
                                               statistics_.statisticViewPtr_->hessians_row_cbegin(statisticIndex),
                                               statistics_.statisticViewPtr_->hessians_row_cend(statisticIndex),
                                               labelIndices_, weight);
                    }

                    /**
                     * @see `IStatisticsSubset::resetSubset`
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
                     * @see `IStatisticsSubset::calculatePrediction`
                     */
                    const IScoreVector& calculatePrediction(bool uncovered, bool accumulated) override final {
                        StatisticVector& sumsOfStatistics = accumulated ? *accumulatedSumVectorPtr_ : sumVector_;

                        if (uncovered) {
                            tmpVector_.difference(totalSumVector_->gradients_cbegin(),
                                                  totalSumVector_->gradients_cend(), totalSumVector_->hessians_cbegin(),
                                                  totalSumVector_->hessians_cend(), labelIndices_,
                                                  sumsOfStatistics.gradients_cbegin(),
                                                  sumsOfStatistics.gradients_cend(), sumsOfStatistics.hessians_cbegin(),
                                                  sumsOfStatistics.hessians_cend());
                            return ruleEvaluationPtr_->calculatePrediction(tmpVector_);
                        }

                        return ruleEvaluationPtr_->calculatePrediction(sumsOfStatistics);
                    }

            };

        protected:

            /**
             * An unique pointer to an object of template type `StatisticView` that stores the gradients and Hessians.
             */
            std::unique_ptr<StatisticView> statisticViewPtr_;

            /**
             * A pointer to an object of template type `RuleEvaluationFactory` to be used for calculating the
             * predictions, as well as corresponding quality scores, of rules.
             */
            const RuleEvaluationFactory* ruleEvaluationFactoryPtr_;

        public:

            /**
             * @param statisticViewPtr      An unique pointer to an object of template type `StatisticView` that
             *                              provides access to the gradients and Hessians
             * @param ruleEvaluationFactory A reference to an object of template type `RuleEvaluationFactory`, to be
             *                              used for calculating the predictions, as well as corresponding quality
             *                              scores, of rules
             */
            AbstractExampleWiseImmutableStatistics(std::unique_ptr<StatisticView> statisticViewPtr,
                                                   const RuleEvaluationFactory& ruleEvaluationFactory)
                : statisticViewPtr_(std::move(statisticViewPtr)), ruleEvaluationFactoryPtr_(&ruleEvaluationFactory) {

            }

            /**
             * @see `IImmutableStatistics::getNumStatistics`
             */
            uint32 getNumStatistics() const override final {
                return statisticViewPtr_->getNumRows();
            }

            /**
             * @see `IImmutableStatistics::getNumLabels`
             */
            uint32 getNumLabels() const override final {
                return statisticViewPtr_->getNumCols();
            }

    };

    /**
     * Provides access to gradients and Hessians that are calculated according to a differentiable loss function that is
     * applied example-wise and are organized as a histogram.
     *
     * @tparam StatisticVector          The type of the vectors that are used to store gradients and Hessians
     * @tparam StatisticView            The type of the view that provides access to the original gradients and Hessians
     * @tparam Histogram                The type of a histogram that stores aggregated gradients and Hessians
     * @tparam RuleEvaluationFactory    The type of the classes that may be used for calculating the predictions, as
     *                                  well as corresponding quality scores, of rules
     */
    template<typename StatisticVector, typename StatisticView, typename Histogram, typename RuleEvaluationFactory>
    class ExampleWiseHistogram final : public AbstractExampleWiseImmutableStatistics<StatisticVector, Histogram,
                                                                                     RuleEvaluationFactory>,
                                       virtual public IHistogram {

        private:

            /**
             * Provides access to a subset of the gradients and Hessians that are stored by an instance of the class
             * `ExampleWiseHistogram`.
             *
             * @tparam T The type of the vector that provides access to the indices of the labels that are included in
             *           the subset
             */
            template<typename T>
            class StatisticsSubset final :
                    public AbstractExampleWiseImmutableStatistics<StatisticVector, Histogram,
                                                                  RuleEvaluationFactory>::template AbstractStatisticsSubset<T> {

                private:

                    const ExampleWiseHistogram& histogram_;

                    std::unique_ptr<StatisticVector> totalCoverableSumVectorPtr_;

                public:

                    /**
                     * @param histogram         A reference to an object of type `ExampleWiseHistogram` that stores the
                     *                          gradients and Hessians
                     * @param totalSumVector    A reference to an object of template type `StatisticVector` that stores
                     *                          the total sums of gradients and Hessians
                     * @param ruleEvaluationPtr An unique pointer to an object of type `IRuleEvaluation` that should be
                     *                          used to calculate the predictions, as well as corresponding quality
                     *                          scores, of rules
                     * @param labelIndices      A reference to an object of template type `T` that provides access to
                     *                          the indices of the labels that are included in the subset
                     */
                    StatisticsSubset(const ExampleWiseHistogram& histogram, const StatisticVector& totalSumVector,
                                     std::unique_ptr<IRuleEvaluation<StatisticVector>> ruleEvaluationPtr,
                                     const T& labelIndices)
                        : AbstractExampleWiseImmutableStatistics<StatisticVector, Histogram,
                                                                 RuleEvaluationFactory>::template AbstractStatisticsSubset<T>(
                              histogram, totalSumVector, std::move(ruleEvaluationPtr), labelIndices),
                          histogram_(histogram) {

                    }

                    /**
                     * @see `IStatisticsSubset::addToMissing`
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
                        totalCoverableSumVectorPtr_->add(originalStatisticView.gradients_row_cbegin(statisticIndex),
                                                         originalStatisticView.gradients_row_cend(statisticIndex),
                                                         originalStatisticView.hessians_row_cbegin(statisticIndex),
                                                         originalStatisticView.hessians_row_cend(statisticIndex),
                                                         -weight);
                    }

            };

            const StatisticView& originalStatisticView_;

            const StatisticVector& totalSumVector_;

        public:

            /**
             * @param originalStatisticView A reference to an object of template type `StatisticView` that provides
             *                              access to the original gradients and Hessians, the histogram was created
             *                              from
             * @param totalSumVector        A reference to an object of template type `StatisticVector` that stores the
             *                              total sums of gradients and Hessians
             * @param ruleEvaluationFactory A reference to an object of type `RuleEvaluationFactory`, to be used for
             *                              calculating the predictions, as well as corresponding quality scores, of
             *                              rules
             * @param numBins               The number of bins in the histogram
             */
            ExampleWiseHistogram(const StatisticView& originalStatisticView, const StatisticVector& totalSumVector,
                                 const RuleEvaluationFactory& ruleEvaluationFactory, uint32 numBins)
                : AbstractExampleWiseImmutableStatistics<StatisticVector, Histogram, RuleEvaluationFactory>(
                      std::make_unique<Histogram>(numBins, originalStatisticView.getNumCols()), ruleEvaluationFactory),
                  originalStatisticView_(originalStatisticView), totalSumVector_(totalSumVector) {

            }

            /**
             * @see `IHistogram::clear`
             */
            void clear() override {
                this->statisticViewPtr_->clear();
            }

            /**
             * @see `IHistogram::addToBin`
             */
            void addToBin(uint32 binIndex, uint32 statisticIndex, float64 weight) override {
                this->statisticViewPtr_->addToRow(binIndex, originalStatisticView_.gradients_row_cbegin(statisticIndex),
                                                  originalStatisticView_.gradients_row_cend(statisticIndex),
                                                  originalStatisticView_.hessians_row_cbegin(statisticIndex),
                                                  originalStatisticView_.hessians_row_cend(statisticIndex), weight);
            }

            /**
             * @see `IImmutableStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
                    const CompleteIndexVector& labelIndices) const override final {
                std::unique_ptr<IRuleEvaluation<StatisticVector>> ruleEvaluationPtr =
                    this->ruleEvaluationFactoryPtr_->create(totalSumVector_, labelIndices);
                return std::make_unique<StatisticsSubset<CompleteIndexVector>>(*this, totalSumVector_,
                                                                               std::move(ruleEvaluationPtr),
                                                                               labelIndices);
            }

            /**
             * @see `IImmutableStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
                    const PartialIndexVector& labelIndices) const override final {
                std::unique_ptr<IRuleEvaluation<StatisticVector>> ruleEvaluationPtr =
                    this->ruleEvaluationFactoryPtr_->create(totalSumVector_, labelIndices);
                return std::make_unique<StatisticsSubset<PartialIndexVector>>(*this, totalSumVector_,
                                                                              std::move(ruleEvaluationPtr),
                                                                              labelIndices);
            }

    };

    /**
     * Provides access to gradients and Hessians that are calculated according to a differentiable loss function that is
     * applied example-wise and allows to update the gradients and Hessians after a new rule has been learned.
     *
     * @tparam LabelMatrix                      The type of the matrix that provides access to the labels of the
     *                                          training examples
     * @tparam StatisticVector                  The type of the vectors that are used to store gradients and Hessians
     * @tparam StatisticView                    The type of the view that provides access to the gradients and Hessians
     * @tparam Histogram                        The type of a histogram that stores aggregated gradients and Hessians
     * @tparam ScoreMatrix                      The type of the matrices that are used to store predicted scores
     * @tparam LossFunction                     The type of the loss function that is used to calculate gradients and Hessians
     * @tparam EvaluationMeasure                The type of the evaluation measure that is used to assess the quality of
     *                                          predictions for a specific statistic
     * @tparam LabelWiseRuleEvaluationFactory   The type of the classes that may be used for calculating the label-wise
     *                                          predictions, as well as corresponding quality scores, of rules
     * @tparam ExampleWiseRuleEvaluationFactory The type of the classes that may be used for calculating the
     *                                          example-wise predictions, as well as corresponding quality scores, of
     *                                          rules
     */
    template<typename LabelMatrix, typename StatisticVector, typename StatisticView, typename Histogram,
             typename ScoreMatrix, typename LossFunction, typename EvaluationMeasure,
             typename ExampleWiseRuleEvaluationFactory, typename LabelWiseRuleEvaluationFactory>
    class AbstractExampleWiseStatistics :
            public AbstractExampleWiseImmutableStatistics<StatisticVector, StatisticView,
                                                          ExampleWiseRuleEvaluationFactory>,
            virtual public IExampleWiseStatistics<ExampleWiseRuleEvaluationFactory, LabelWiseRuleEvaluationFactory> {

        private:


            /**
             * Provides access to a subset of the gradients and Hessians that are stored by an instance of the class
             * `AbstractExampleWiseStatistics`.
             *
             * @tparam T The type of the vector that provides access to the indices of the labels that are included in
             *           the subset
             */
            template<typename T>
            class StatisticsSubset final :
                    public AbstractExampleWiseImmutableStatistics<StatisticVector, StatisticView,
                                                                  ExampleWiseRuleEvaluationFactory>::template AbstractStatisticsSubset<T> {

                private:

                    std::unique_ptr<StatisticVector> totalCoverableSumVectorPtr_;

                public:

                    /**
                     * @param statistics        A reference to an object of type
                     *                          `AbstractExampleWiseImmutableStatistics` that stores the gradients and
                     *                          Hessians
                     * @param totalSumVector    A reference to an object of template type `StatisticVector` that stores
                     *                          the total sums of gradients and Hessians
                     * @param ruleEvaluationPtr An unique pointer to an object of type `IRuleEvaluation` that should be
                     *                          used to calculate the predictions, as well as corresponding quality
                     *                          scores, of rules
                     * @param labelIndices      A reference to an object of template type `T` that provides access to
                     *                          the indices of the labels that are included in the subset
                     */
                    StatisticsSubset(const AbstractExampleWiseStatistics& statistics,
                                     const StatisticVector& totalSumVector,
                                     std::unique_ptr<IRuleEvaluation<StatisticVector>> ruleEvaluationPtr,
                                     const T& labelIndices)
                        : AbstractExampleWiseImmutableStatistics<StatisticVector, StatisticView,
                                                                 ExampleWiseRuleEvaluationFactory>::template AbstractStatisticsSubset<T>(
                              statistics, totalSumVector, std::move(ruleEvaluationPtr), labelIndices) {

                    }

                    /**
                     * @see `IStatisticsSubset::addToMissing`
                     */
                    void addToMissing(uint32 statisticIndex, float64 weight) override {
                        // Create a vector for storing the totals sums of gradients and Hessians, if necessary...
                        if (!totalCoverableSumVectorPtr_) {
                            totalCoverableSumVectorPtr_ = std::make_unique<StatisticVector>(*this->totalSumVector_);
                            this->totalSumVector_ = totalCoverableSumVectorPtr_.get();
                        }

                        // Subtract the gradients and Hessians of the example at the given index (weighted by the given
                        // weight) from the total sums of gradients and Hessians...
                        const StatisticView& statisticView = this->getStatisticView();
                        totalCoverableSumVectorPtr_->add(statisticView.gradients_row_cbegin(statisticIndex),
                                                         statisticView.gradients_row_cend(statisticIndex),
                                                         statisticView.hessians_row_cbegin(statisticIndex),
                                                         statisticView.hessians_row_cend(statisticIndex), -weight);
                    }

            };

            std::unique_ptr<StatisticVector> totalSumVectorPtr_;

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
             *                              `ExampleWiseRuleEvaluationFactory`, to be used for calculating the
             *                              predictions, as well as corresponding quality scores, of rules
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
                : AbstractExampleWiseImmutableStatistics<StatisticVector, StatisticView,
                                                         ExampleWiseRuleEvaluationFactory>(
                      std::move(statisticViewPtr), ruleEvaluationFactory),
                  totalSumVectorPtr_(std::make_unique<StatisticVector>(this->statisticViewPtr_->getNumCols())),
                  lossPtr_(std::move(lossPtr)), evaluationMeasurePtr_(std::move(evaluationMeasurePtr)),
                  labelMatrix_(labelMatrix), scoreMatrixPtr_(std::move(scoreMatrixPtr)) {

            }

            /**
             * @see `IExampleWiseStatistics::setRuleEvaluationFactory`
             */
            void setRuleEvaluationFactory(const ExampleWiseRuleEvaluationFactory& ruleEvaluationFactory) override final {
                this->ruleEvaluationFactoryPtr_ = &ruleEvaluationFactory;
            }

            /**
             * @see `IStatistics::resetSampledStatistics`
             */
            void resetSampledStatistics() override final {
                // This function is equivalent to the function `resetCoveredStatistics`...
                this->resetCoveredStatistics();
            }

            /**
             * @see `IStatistics::addSampledStatistic`
             */
            void addSampledStatistic(uint32 statisticIndex, float64 weight) override final {
                // This function is equivalent to the function `updateCoveredStatistic`...
                this->updateCoveredStatistic(statisticIndex, weight, false);
            }

            /**
             * @see `IStatistics::resetCoveredStatistics`
             */
            void resetCoveredStatistics() override final {
                totalSumVectorPtr_->clear();
            }

            /**
             * @see `IStatistics::updateCoveredStatistic`
             */
            void updateCoveredStatistic(uint32 statisticIndex, float64 weight, bool remove) override final {
                float64 signedWeight = remove ? -weight : weight;
                totalSumVectorPtr_->add(this->statisticViewPtr_->gradients_row_cbegin(statisticIndex),
                                        this->statisticViewPtr_->gradients_row_cend(statisticIndex),
                                        this->statisticViewPtr_->hessians_row_cbegin(statisticIndex),
                                        this->statisticViewPtr_->hessians_row_cend(statisticIndex), signedWeight);
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
             * @see `IStatistics::createHistogram`
             */
            std::unique_ptr<IHistogram> createHistogram(uint32 numBins) const override final {
                return std::make_unique<ExampleWiseHistogram<StatisticVector, StatisticView, Histogram,
                                                             ExampleWiseRuleEvaluationFactory>>(
                    *this->statisticViewPtr_, *totalSumVectorPtr_, *this->ruleEvaluationFactoryPtr_, numBins);
            }

            /**
             * @see `IImmutableStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
                    const CompleteIndexVector& labelIndices) const override final {
                std::unique_ptr<IRuleEvaluation<StatisticVector>> ruleEvaluationPtr =
                    this->ruleEvaluationFactoryPtr_->create(*totalSumVectorPtr_, labelIndices);
                return std::make_unique<StatisticsSubset<CompleteIndexVector>>(*this, *totalSumVectorPtr_,
                                                                               std::move(ruleEvaluationPtr),
                                                                               labelIndices);
            }

            /**
             * @see `IImmutableStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
                    const PartialIndexVector& labelIndices) const override final {
                std::unique_ptr<IRuleEvaluation<StatisticVector>> ruleEvaluationPtr =
                    this->ruleEvaluationFactoryPtr_->create(*totalSumVectorPtr_, labelIndices);
                return std::make_unique<StatisticsSubset<PartialIndexVector>>(*this, *totalSumVectorPtr_,
                                                                              std::move(ruleEvaluationPtr),
                                                                              labelIndices);
            }

    };

}
