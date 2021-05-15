#include "boosting/statistics/statistics_example_wise.hpp"


namespace boosting {

    template<class Prediction, class LabelMatrix, class StatisticView, class ScoreMatrix>
    void applyPredictionInternally(uint32 statisticIndex, const Prediction& prediction, const LabelMatrix& labelMatrix,
                                   StatisticView& statisticView, ScoreMatrix& scoreMatrix,
                                   const IExampleWiseLoss& lossFunction) {
        // Update the scores that are currently predicted for the example at the given index...
        scoreMatrix.addToRowFromSubset(statisticIndex, prediction.scores_cbegin(), prediction.scores_cend(),
                                       prediction.indices_cbegin(), prediction.indices_cend());

        // Update the gradients and Hessians for the example at the given index...
        lossFunction.updateExampleWiseStatistics(statisticIndex, labelMatrix, scoreMatrix, statisticView);
    }

    /**
     * An abstract base class for all statistics that provide access to gradients and Hessians that are calculated
     * according to a differentiable loss function that is applied example-wise.
     *
     * @tparam StatisticVector  The type of the vectors that are used to store gradients and Hessians
     * @tparam StatisticView    The type of the view that provides access to the gradients and Hessians
     * @tparam StatisticMatrix  The type of the matrix that stores the gradients and Hessians
     * @tparam ScoreMatrix      The type of the matrices that are used to store predicted scores
     */
    template<class StatisticVector, class StatisticView, class StatisticMatrix, class ScoreMatrix>
    class AbstractExampleWiseImmutableStatistics : virtual public IImmutableStatistics {

        protected:

            /**
             * Provides access to a subset of the gradients and Hessians that are stored by an instance of the class
             * `AbstractExampleWiseImmutableStatistics`.
             *
             * @tparam T The type of the vector that provides access to the indices of the labels that are included in
             *           the subset
             */
            template<class T>
            class StatisticsSubset final : public IStatisticsSubset {

                private:

                    const AbstractExampleWiseImmutableStatistics& statistics_;

                    const StatisticVector* totalSumVector_;

                    std::unique_ptr<IExampleWiseRuleEvaluation> ruleEvaluationPtr_;

                    const T& labelIndices_;

                    StatisticVector sumVector_;

                    StatisticVector* accumulatedSumVector_;

                    StatisticVector* totalCoverableSumVector_;

                    StatisticVector tmpVector_;

                public:

                    /**
                     * @param statistics        A reference to an object of type
                     *                          `AbstractExampleWiseImmutableStatistics` that stores the gradients and
                     *                          Hessians
                     * @param totalSumVector    A pointer to an object of template type `StatisticVector` that stores
                     *                          the total sums of gradients and Hessians
                     * @param ruleEvaluationPtr An unique pointer to an object of type `IExampleWiseRuleEvaluation` that
                     *                          should be used to calculate the predictions, as well as corresponding
                     *                          quality scores, of rules
                     * @param labelIndices      A reference to an object of template type `T` that provides access to
                     *                          the indices of the labels that are included in the subset
                     */
                    StatisticsSubset(const AbstractExampleWiseImmutableStatistics& statistics,
                                     const StatisticVector* totalSumVector,
                                     std::unique_ptr<IExampleWiseRuleEvaluation> ruleEvaluationPtr,
                                     const T& labelIndices)
                        : statistics_(statistics), totalSumVector_(totalSumVector),
                          ruleEvaluationPtr_(std::move(ruleEvaluationPtr)), labelIndices_(labelIndices),
                          sumVector_(StatisticVector(labelIndices.getNumElements(), true)),
                          accumulatedSumVector_(nullptr), totalCoverableSumVector_(nullptr),
                          tmpVector_(StatisticVector(labelIndices.getNumElements())) {

                    }

                    ~StatisticsSubset() {
                        delete accumulatedSumVector_;
                        delete totalCoverableSumVector_;
                    }

                    void addToMissing(uint32 statisticIndex, float64 weight) override {
                        // Create a vector for storing the totals sums of gradients and Hessians, if necessary...
                        if (totalCoverableSumVector_ == nullptr) {
                            totalCoverableSumVector_ = new StatisticVector(*totalSumVector_);
                            totalSumVector_ = totalCoverableSumVector_;
                        }

                        // Subtract the gradients and Hessians of the example at the given index (weighted by the given
                        // weight) from the total sums of gradients and Hessians...
                        totalCoverableSumVector_->add(
                            statistics_.statisticViewPtr_->gradients_row_cbegin(statisticIndex),
                            statistics_.statisticViewPtr_->gradients_row_cend(statisticIndex),
                            statistics_.statisticViewPtr_->hessians_row_cbegin(statisticIndex),
                            statistics_.statisticViewPtr_->hessians_row_cend(statisticIndex), -weight);
                    }

                    void addToSubset(uint32 statisticIndex, float64 weight) override {
                        sumVector_.addToSubset(statistics_.statisticViewPtr_->gradients_row_cbegin(statisticIndex),
                                               statistics_.statisticViewPtr_->gradients_row_cend(statisticIndex),
                                               statistics_.statisticViewPtr_->hessians_row_cbegin(statisticIndex),
                                               statistics_.statisticViewPtr_->hessians_row_cend(statisticIndex),
                                               labelIndices_, weight);
                    }

                    void resetSubset() override {
                        // Create a vector for storing the accumulated sums of gradients and Hessians, if necessary...
                        if (accumulatedSumVector_ == nullptr) {
                            uint32 numPredictions = labelIndices_.getNumElements();
                            accumulatedSumVector_ = new StatisticVector(numPredictions, true);
                        }

                        // Reset the sum of gradients and Hessians to zero and add it to the accumulated sums of
                        // gradients and Hessians...
                        accumulatedSumVector_->add(sumVector_.gradients_cbegin(), sumVector_.gradients_cend(),
                                                   sumVector_.hessians_cbegin(), sumVector_.hessians_cend());
                        sumVector_.setAllToZero();
                    }

                    const ILabelWiseScoreVector& calculateLabelWisePrediction(bool uncovered,
                                                                              bool accumulated) override {
                        const StatisticVector& sumsOfStatistics = accumulated ? *accumulatedSumVector_ : sumVector_;

                        if (uncovered) {
                            tmpVector_.difference(totalSumVector_->gradients_cbegin(),
                                                  totalSumVector_->gradients_cend(), totalSumVector_->hessians_cbegin(),
                                                  totalSumVector_->hessians_cend(), labelIndices_,
                                                  sumsOfStatistics.gradients_cbegin(),
                                                  sumsOfStatistics.gradients_cend(), sumsOfStatistics.hessians_cbegin(),
                                                  sumsOfStatistics.hessians_cend());
                            return ruleEvaluationPtr_->calculateLabelWisePrediction(tmpVector_);
                        }

                        return ruleEvaluationPtr_->calculateLabelWisePrediction(sumsOfStatistics);
                    }

                    const IScoreVector& calculateExampleWisePrediction(bool uncovered, bool accumulated) override {
                        StatisticVector& sumsOfStatistics = accumulated ? *accumulatedSumVector_ : sumVector_;

                        if (uncovered) {
                            tmpVector_.difference(totalSumVector_->gradients_cbegin(),
                                                  totalSumVector_->gradients_cend(), totalSumVector_->hessians_cbegin(),
                                                  totalSumVector_->hessians_cend(), labelIndices_,
                                                  sumsOfStatistics.gradients_cbegin(),
                                                  sumsOfStatistics.gradients_cend(), sumsOfStatistics.hessians_cbegin(),
                                                  sumsOfStatistics.hessians_cend());
                            return ruleEvaluationPtr_->calculateExampleWisePrediction(tmpVector_);
                        }

                        return ruleEvaluationPtr_->calculateExampleWisePrediction(sumsOfStatistics);
                    }

            };

            /**
             * The type of a `StatisticsSubset` that corresponds to all available labels.
             */
            typedef StatisticsSubset<FullIndexVector> FullSubset;

            /**
             * The type of a `StatisticsSubset` that corresponds to a subset of the available labels.
             */
            typedef StatisticsSubset<PartialIndexVector> PartialSubset;

        private:

            uint32 numStatistics_;

            uint32 numLabels_;

        protected:

            /**
             * An unique pointer to an object of template type `StatisticView` that stores the gradients and Hessians.
             */
            std::unique_ptr<StatisticView> statisticViewPtr_;

            /**
             * A shared pointer to an object of type `IExampleWiseRuleEvaluationFactory` to be used for calculating the
             * predictions, as well as corresponding quality scores, of rules.
             */
            std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr_;

        public:

            /**
             * @param statisticViewPtr          An unique pointer to an object of template type `StatisticView` that
             *                                  provides access to the gradients and Hessians
             * @param ruleEvaluationFactoryPtr  A shared pointer to an object of type
             *                                  `IExampleWiseRuleEvaluationFactory`, to be used for calculating the
             *                                  predictions, as well as corresponding quality scores, of rules
             */
            AbstractExampleWiseImmutableStatistics(
                    std::unique_ptr<StatisticView> statisticViewPtr,
                    std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr)
                : numStatistics_(statisticViewPtr->getNumRows()), numLabels_(statisticViewPtr->getNumCols()),
                  statisticViewPtr_(std::move(statisticViewPtr)), ruleEvaluationFactoryPtr_(ruleEvaluationFactoryPtr) {

            }

            uint32 getNumStatistics() const override final {
                return numStatistics_;
            }

            uint32 getNumLabels() const override final {
                return numLabels_;
            }

    };

    /**
     * Provides access to gradients and Hessians that are calculated according to a differentiable loss function that is
     * applied example-wise and are organized as a histogram.
     *
     * @tparam StatisticVector  The type of the vectors that are used to store gradients and Hessians
     * @tparam StatisticView    The type of the view that provides access to the gradients and Hessians
     * @tparam StatisticMatrix  The type of the matrix that stores the gradients and Hessians
     * @tparam ScoreMatrix      The type of the matrices that are used to store predicted scores
     */
    template<class StatisticVector, class StatisticView, class StatisticMatrix, class ScoreMatrix>
    class ExampleWiseHistogram final : public AbstractExampleWiseImmutableStatistics<StatisticVector, StatisticView,
                                                                                     StatisticMatrix, ScoreMatrix>,
                                       virtual public IHistogram {

        private:

            const StatisticView& statisticView_;

            const StatisticVector* totalSumVector_;

        public:

            /**
             * @param statisticView             A reference to an object of template type `StatisticView` that provides
             *                                  access to the original gradients and Hessians, the histogram was created
             *                                  from
             * @param totalSumVector            A pointer to an object of template type `StatisticVector` that stores
             *                                  the total sums of gradients and Hessians
             * @param ruleEvaluationFactoryPtr  A shared pointer to an object of type
             *                                  `IExampleWiseRuleEvaluationFactory`, to be used for calculating the
             *                                  predictions, as well as corresponding quality scores, of rules
             * @param numBins                   The number of bins in the histogram
             */
            ExampleWiseHistogram(const StatisticView& statisticView, const StatisticVector* totalSumVector,
                                 std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
                                 uint32 numBins)
                : AbstractExampleWiseImmutableStatistics<StatisticVector, StatisticView, StatisticMatrix, ScoreMatrix>(
                      std::make_unique<StatisticMatrix>(numBins, statisticView.getNumCols()), ruleEvaluationFactoryPtr),
                  statisticView_(statisticView), totalSumVector_(totalSumVector) {

            }

            void setAllToZero() override {
                this->statisticViewPtr_->setAllToZero();
            }

            void addToBin(uint32 binIndex, uint32 statisticIndex, uint32 weight) override {
                this->statisticViewPtr_->addToRow(binIndex, statisticView_.gradients_row_cbegin(statisticIndex),
                                                  statisticView_.gradients_row_cend(statisticIndex),
                                                  statisticView_.hessians_row_cbegin(statisticIndex),
                                                  statisticView_.hessians_row_cend(statisticIndex), weight);
            }

            std::unique_ptr<IStatisticsSubset> createSubset(const FullIndexVector& labelIndices) const override final {
                std::unique_ptr<IExampleWiseRuleEvaluation> ruleEvaluationPtr =
                    this->ruleEvaluationFactoryPtr_->create(labelIndices);
                return std::make_unique<typename ExampleWiseHistogram::FullSubset>(*this, totalSumVector_,
                                                                                   std::move(ruleEvaluationPtr),
                                                                                   labelIndices);
            }

            std::unique_ptr<IStatisticsSubset> createSubset(
                    const PartialIndexVector& labelIndices) const override final {
                std::unique_ptr<IExampleWiseRuleEvaluation> ruleEvaluationPtr =
                    this->ruleEvaluationFactoryPtr_->create(labelIndices);
                return std::make_unique<typename ExampleWiseHistogram::PartialSubset>(*this, totalSumVector_,
                                                                                      std::move(ruleEvaluationPtr),
                                                                                      labelIndices);
            }

    };

    /**
     * Provides access to gradients and Hessians that are calculated according to a differentiable loss function that is
     * applied example-wise and allows to update the gradients and Hessians after a new rule has been learned.
     *
     * @tparam LabelMatrix      The type of the matrix that provides access to the labels of the training examples
     * @tparam StatisticVector  The type of the vectors that are used to store gradients and Hessians
     * @tparam StatisticView    The type of the view that provides access to the gradients and Hessians
     * @tparam StatisticMatrix  The type of the matrix that stores the gradients and Hessians
     * @tparam ScoreMatrix      The type of the matrices that are used to store predicted scores
     */
    template<class LabelMatrix, class StatisticVector, class StatisticView, class StatisticMatrix, class ScoreMatrix>
    class AbstractExampleWiseStatistics : public AbstractExampleWiseImmutableStatistics<StatisticVector, StatisticView,
                                                                                        StatisticMatrix, ScoreMatrix>,
                                  virtual public IExampleWiseStatistics {

        private:

            std::unique_ptr<StatisticVector> totalSumVectorPtr_;

            std::shared_ptr<IExampleWiseLoss> lossFunctionPtr_;

            const LabelMatrix& labelMatrix_;

            std::unique_ptr<ScoreMatrix> scoreMatrixPtr_;

        public:

            /**
             * @param lossFunctionPtr           A shared pointer to an object of type `IExampleWiseLoss`, representing
             *                                  the loss function to be used for calculating gradients and Hessians
             * @param ruleEvaluationFactoryPtr  A shared pointer to an object of type
             *                                  `IExampleWiseRuleEvaluationFactory`, to be used for calculating the
             *                                  predictions, as well as corresponding quality scores, of rules
             * @param labelMatrix               A reference to an object of template type `LabelMatrix` that provides
             *                                  access to the labels of the training examples
             * @param statisticViewPtr          An unique pointer to an object of template type `StatisticView` that
             *                                  provides access to the gradients and Hessians
             * @param scoreMatrixPtr            An unique pointer to an object of template type `ScoreMatrix` that
             *                                  stores the currently predicted scores
             */
            AbstractExampleWiseStatistics(std::shared_ptr<IExampleWiseLoss> lossFunctionPtr,
                                          std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
                                          const LabelMatrix& labelMatrix,
                                          std::unique_ptr<StatisticView> statisticViewPtr,
                                          std::unique_ptr<ScoreMatrix> scoreMatrixPtr)
                : AbstractExampleWiseImmutableStatistics<StatisticVector, StatisticView, StatisticMatrix, ScoreMatrix>(
                      std::move(statisticViewPtr), ruleEvaluationFactoryPtr),
                  totalSumVectorPtr_(std::make_unique<StatisticVector>(this->statisticViewPtr_->getNumCols())),
                  lossFunctionPtr_(lossFunctionPtr), labelMatrix_(labelMatrix),
                  scoreMatrixPtr_(std::move(scoreMatrixPtr)) {

            }

            /**
             * @see `IExampleWiseStatistics::setRuleEvaluationFactory`
             */
            void setRuleEvaluationFactory(
                    std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr) override {
                this->ruleEvaluationFactoryPtr_ = ruleEvaluationFactoryPtr;
            }

            /**
             * @see `IStatistics::resetSampledStatistics`
             */
            void resetSampledStatistics() override {
                // This function is equivalent to the function `resetCoveredStatistics`...
                this->resetCoveredStatistics();
            }

            /**
             * @see `IStatistics::addSampledStatistic`
             */
            void addSampledStatistic(uint32 statisticIndex, float64 weight) override {
                // This function is equivalent to the function `updateCoveredStatistic`...
                this->updateCoveredStatistic(statisticIndex, weight, false);
            }

            /**
             * @see `IStatistics::resetCoveredStatistics`
             */
            void resetCoveredStatistics() override {
                totalSumVectorPtr_->setAllToZero();
            }

            /**
             * @see `IStatistics::updateCoveredStatistic`
             */
            void updateCoveredStatistic(uint32 statisticIndex, float64 weight, bool remove) override {
                float64 signedWeight = remove ? -weight : weight;
                totalSumVectorPtr_->add(this->statisticViewPtr_->gradients_row_cbegin(statisticIndex),
                                        this->statisticViewPtr_->gradients_row_cend(statisticIndex),
                                        this->statisticViewPtr_->hessians_row_cbegin(statisticIndex),
                                        this->statisticViewPtr_->hessians_row_cend(statisticIndex), signedWeight);
            }

            /**
             * @see `IStatistics::applyPrediction`
             */
            void applyPrediction(uint32 statisticIndex, const FullPrediction& prediction) override {
                applyPredictionInternally<FullPrediction, LabelMatrix, StatisticView, ScoreMatrix>(
                    statisticIndex, prediction, labelMatrix_, *this->statisticViewPtr_, *scoreMatrixPtr_,
                    *lossFunctionPtr_);
            }

            /**
             * @see `IStatistics::applyPrediction`
             */
            void applyPrediction(uint32 statisticIndex, const PartialPrediction& prediction) override {
                applyPredictionInternally<PartialPrediction, LabelMatrix, StatisticView, ScoreMatrix>(
                    statisticIndex, prediction, labelMatrix_, *this->statisticViewPtr_, *scoreMatrixPtr_,
                    *lossFunctionPtr_);
            }

            /**
             * @see `IStatistics::evaluatePrediction`
             */
            float64 evaluatePrediction(uint32 statisticIndex, const IEvaluationMeasure& measure) const override {
                return measure.evaluate(statisticIndex, labelMatrix_, *scoreMatrixPtr_);
            }

            /**
             * @see `IStatistics::createHistogram`
             */
            std::unique_ptr<IHistogram> createHistogram(uint32 numBins) const override {
                return std::make_unique<ExampleWiseHistogram<StatisticVector, StatisticView, StatisticMatrix,
                                                             ScoreMatrix>>(*this->statisticViewPtr_,
                                                                           totalSumVectorPtr_.get(),
                                                                           this->ruleEvaluationFactoryPtr_, numBins);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const FullIndexVector& labelIndices) const override final {
                std::unique_ptr<IExampleWiseRuleEvaluation> ruleEvaluationPtr =
                    this->ruleEvaluationFactoryPtr_->create(labelIndices);
                return std::make_unique<typename AbstractExampleWiseStatistics::FullSubset>(
                    *this, totalSumVectorPtr_.get(), std::move(ruleEvaluationPtr), labelIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
                    const PartialIndexVector& labelIndices) const override final {
                std::unique_ptr<IExampleWiseRuleEvaluation> ruleEvaluationPtr =
                    this->ruleEvaluationFactoryPtr_->create(labelIndices);
                return std::make_unique<typename AbstractExampleWiseStatistics::PartialSubset>(
                    *this, totalSumVectorPtr_.get(), std::move(ruleEvaluationPtr), labelIndices);
            }

    };

}
