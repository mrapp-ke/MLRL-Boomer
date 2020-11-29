#include "statistics_example_wise.h"
#include "../data/matrix_dense_numeric.h"
#include "../data/matrix_dense_example_wise.h"
#include "../data/vector_dense_example_wise.h"

using namespace boosting;


/**
 * Provides access to gradients and Hessians that are calculated according to a differentiable loss function that is
 * applied example-wise.
 *
 * @tparam StatisticVector  The type of the vectors that are used to store gradients and Hessians
 * @tparam StatisticMatrix  The type of the matrices that are used to store gradients and Hessians
 * @tparam ScoreMatrix      The type of the matrices that are used to store predicted scores
 */
template<class StatisticVector, class StatisticMatrix, class ScoreMatrix>
class ExampleWiseHistogram : virtual public IHistogram {

    private:

        /**
         * Provides access to a subset of the gradients and Hessians that are stored by an instance of the class
         * `ExampleWiseStatistics`.
         *
         * @tparam T The type of the vector that provides access to the indices of the labels that are included in the
         *           subset
         */
        template<class T>
        class StatisticsSubset : public IStatisticsSubset {

            private:

                const ExampleWiseHistogram& histogram_;

                std::unique_ptr<IExampleWiseRuleEvaluation> ruleEvaluationPtr_;

                const T& labelIndices_;

                StatisticVector sumVector_;

                StatisticVector* accumulatedSumVector_;

                const StatisticVector* totalSumVector_;

                StatisticVector* totalCoverableSumVector_;

                StatisticVector tmpVector_;

            public:

                /**
                 * @param histogram         A reference to an object of type `ExampleWiseHistogram` that stores the
                 *                          gradients and Hessians
                 * @param ruleEvaluationPtr An unique pointer to an object of type `IExampleWiseRuleEvaluation` that
                 *                          should be used to calculate the predictions, as well as corresponding
                 *                          quality scores, of rules
                 * @param labelIndices      A reference to an object of template type `T` that provides access to the
                 *                          indices of the labels that are included in the subset
                 */
                StatisticsSubset(const ExampleWiseHistogram& histogram,
                                 std::unique_ptr<IExampleWiseRuleEvaluation> ruleEvaluationPtr, const T& labelIndices)
                    : histogram_(histogram), ruleEvaluationPtr_(std::move(ruleEvaluationPtr)),
                      labelIndices_(labelIndices), sumVector_(StatisticVector(labelIndices.getNumElements(), true)),
                      totalSumVector_(histogram.totalSumVectorPtr_.get()),
                      tmpVector_(StatisticVector(labelIndices.getNumElements())) {
                    accumulatedSumVector_ = nullptr;
                    totalCoverableSumVector_ = nullptr;
                }

                ~StatisticsSubset() {
                    delete accumulatedSumVector_;
                    delete totalCoverableSumVector_;
                }

                void addToMissing(uint32 statisticIndex, uint32 weight) override {
                    // Create a vector for storing the totals sums of gradients and Hessians, if necessary...
                    if (totalCoverableSumVector_ == nullptr) {
                        totalCoverableSumVector_ = new StatisticVector(*totalSumVector_);
                        totalSumVector_ = totalCoverableSumVector_;
                    }

                    // Subtract the gradients and Hessians of the example at the given index (weighted by the given
                    // weight) from the total sums of gradients and Hessians...
                    totalCoverableSumVector_->subtract(
                        histogram_.statisticMatrixPtr_->gradients_row_cbegin(statisticIndex),
                        histogram_.statisticMatrixPtr_->gradients_row_cend(statisticIndex),
                        histogram_.statisticMatrixPtr_->hessians_row_cbegin(statisticIndex),
                        histogram_.statisticMatrixPtr_->hessians_row_cend(statisticIndex), weight);
                }

                void addToSubset(uint32 statisticIndex, uint32 weight) override {
                    sumVector_.addToSubset(histogram_.statisticMatrixPtr_->gradients_row_cbegin(statisticIndex),
                                           histogram_.statisticMatrixPtr_->gradients_row_cend(statisticIndex),
                                           histogram_.statisticMatrixPtr_->hessians_row_cbegin(statisticIndex),
                                           histogram_.statisticMatrixPtr_->hessians_row_cend(statisticIndex),
                                           labelIndices_, weight);
                }

                void resetSubset() override {
                    // Create a vector for storing the accumulated sums of gradients and Hessians, if necessary...
                    if (accumulatedSumVector_ == nullptr) {
                        uint32 numPredictions = labelIndices_.getNumElements();
                        accumulatedSumVector_ = new StatisticVector(numPredictions, true);
                    }

                    // Reset the sum of gradients and Hessians to zero and add it to the accumulated sums of gradients
                    // and Hessians...
                    accumulatedSumVector_->add(sumVector_.gradients_cbegin(), sumVector_.gradients_cend(),
                                               sumVector_.hessians_cbegin(), sumVector_.hessians_cend());
                    sumVector_.setAllToZero();
                }

                const DenseLabelWiseScoreVector& calculateLabelWisePrediction(bool uncovered,
                                                                                 bool accumulated) override {
                    const StatisticVector& sumsOfStatistics = accumulated ? *accumulatedSumVector_ : sumVector_;

                    if (uncovered) {
                        tmpVector_.difference(totalSumVector_->gradients_cbegin(), totalSumVector_->gradients_cend(),
                                              totalSumVector_->hessians_cbegin(), totalSumVector_->hessians_cend(),
                                              labelIndices_, sumsOfStatistics.gradients_cbegin(),
                                              sumsOfStatistics.gradients_cend(), sumsOfStatistics.hessians_cbegin(),
                                              sumsOfStatistics.hessians_cend());
                        return ruleEvaluationPtr_->calculateLabelWisePrediction(tmpVector_);
                    }

                    return ruleEvaluationPtr_->calculateLabelWisePrediction(sumsOfStatistics);
                }

                const DenseScoreVector& calculateExampleWisePrediction(bool uncovered, bool accumulated) override {
                    StatisticVector& sumsOfStatistics = accumulated ? *accumulatedSumVector_ : sumVector_;

                    if (uncovered) {
                        tmpVector_.difference(totalSumVector_->gradients_cbegin(), totalSumVector_->gradients_cend(),
                                              totalSumVector_->hessians_cbegin(), totalSumVector_->hessians_cend(),
                                              labelIndices_, sumsOfStatistics.gradients_cbegin(),
                                              sumsOfStatistics.gradients_cend(), sumsOfStatistics.hessians_cbegin(),
                                              sumsOfStatistics.hessians_cend());
                        return ruleEvaluationPtr_->calculateExampleWisePrediction(tmpVector_);
                    }

                    return ruleEvaluationPtr_->calculateExampleWisePrediction(sumsOfStatistics);
                }

        };

        uint32 numStatistics_;

        uint32 numLabels_;

    protected:


        std::unique_ptr<StatisticMatrix> statisticMatrixPtr_;

        std::unique_ptr<StatisticVector> totalSumVectorPtr_;

        std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr_;

    public:

        /**
         * @param statisticMatrixPtr        An unique pointer to an object of template type `StatisticMatrix` that
         *                                  stores the gradients and Hessians
         * @param totalSumVectorPtr         An unique pointer to an object of template type `StatisticVector` that
         *                                  stores the total sums of gradients and Hessians
         * @param ruleEvaluationFactoryPtr  A shared pointer to an object of type `IExampleWiseRuleEvaluationFactory`,
         *                                  to be used for calculating the predictions, as well as corresponding quality
         *                                  scores, of rules
         */
        ExampleWiseHistogram(std::unique_ptr<StatisticMatrix> statisticMatrixPtr,
                             std::unique_ptr<StatisticVector> totalSumVectorPtr,
                             std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr)
            : numStatistics_(statisticMatrixPtr->getNumRows()), numLabels_(statisticMatrixPtr->getNumCols()),
              statisticMatrixPtr_(std::move(statisticMatrixPtr)), totalSumVectorPtr_(std::move(totalSumVectorPtr)),
              ruleEvaluationFactoryPtr_(ruleEvaluationFactoryPtr) {

        }

        uint32 getNumStatistics() const override {
            return numStatistics_;
        }

        uint32 getNumLabels() const override {
            return numLabels_;
        }

        std::unique_ptr<IStatisticsSubset> createSubset(const FullIndexVector& labelIndices) const override {
            std::unique_ptr<IExampleWiseRuleEvaluation> ruleEvaluationPtr =
                ruleEvaluationFactoryPtr_->create(labelIndices);
            return std::make_unique<StatisticsSubset<FullIndexVector>>(*this, std::move(ruleEvaluationPtr),
                                                                       labelIndices);
        }

        std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& labelIndices) const override {
            std::unique_ptr<IExampleWiseRuleEvaluation> ruleEvaluationPtr =
                ruleEvaluationFactoryPtr_->create(labelIndices);
            return std::make_unique<StatisticsSubset<PartialIndexVector>>(*this, std::move(ruleEvaluationPtr),
                                                                          labelIndices);
        }

};

/**
 * Provides access to gradients and Hessians that are calculated according to a differentiable loss function that is
 * applied example-wise and allows to update the gradients and Hessians after a new rule has been learned.
 *
 * @tparam StatisticVector  The type of the vectors that are used to store gradients and Hessians
 * @tparam StatisticMatrix  The type of the matrices that are used to store gradients and Hessians
 * @tparam ScoreMatrix      The type of the matrices that are used to store predicted scores
 */
template<class StatisticVector, class StatisticMatrix, class ScoreMatrix>
class ExampleWiseStatistics : public ExampleWiseHistogram<StatisticVector, StatisticMatrix, ScoreMatrix>,
                              virtual public IExampleWiseStatistics {

    private:

        /**
         * Allows to build a histogram based on the gradients and Hessians that are stored by an instance of the class
         * `ExampleWiseStatistics`.
         */
        class HistogramBuilder : public IHistogramBuilder {

            private:

                const ExampleWiseStatistics& statistics_;

                std::unique_ptr<StatisticMatrix> statisticMatrixPtr_;

            public:

            /**
             * @param statistics    A reference to an object of type `ExampleWiseStatistics` that stores the gradients
             *                      and Hessians
             * @param numBins       The number of bins, the histogram should consist of
             */
            HistogramBuilder(const ExampleWiseStatistics& statistics, uint32 numBins)
                : statistics_(statistics),
                  statisticMatrixPtr_(std::make_unique<StatisticMatrix>(numBins, statistics.getNumLabels(), true)) {

            }

            void onBinUpdate(uint32 binIndex, uint32 originalIndex, float32 value) override {
                statisticMatrixPtr_->addToRow(binIndex,
                                              statistics_.statisticMatrixPtr_->gradients_row_cbegin(originalIndex),
                                              statistics_.statisticMatrixPtr_->gradients_row_cend(originalIndex),
                                              statistics_.statisticMatrixPtr_->hessians_row_cbegin(originalIndex),
                                              statistics_.statisticMatrixPtr_->hessians_row_cend(originalIndex));
            }

            std::unique_ptr<IHistogram> build() override {
                std::unique_ptr<StatisticVector> totalSumVectorPtr =
                    std::make_unique<StatisticVector>(statistics_.getNumLabels(), true);
                uint32 numBins = statisticMatrixPtr_->getNumRows();

                for (uint32 i = 0; i < numBins; i++) {
                    totalSumVectorPtr->add(statisticMatrixPtr_->gradients_row_cbegin(i),
                                           statisticMatrixPtr_->gradients_row_cend(i),
                                           statisticMatrixPtr_->hessians_row_cbegin(i),
                                           statisticMatrixPtr_->hessians_row_cend(i));
                }

                return std::make_unique<ExampleWiseHistogram<StatisticVector, StatisticMatrix, ScoreMatrix>>(
                    std::move(statisticMatrixPtr_), std::move(totalSumVectorPtr),
                    statistics_.ruleEvaluationFactoryPtr_);
            }

        };

        std::shared_ptr<IExampleWiseLoss> lossFunctionPtr_;

        std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr_;

        std::unique_ptr<ScoreMatrix> scoreMatrixPtr_;

        template<class T>
        void applyPredictionInternally(uint32 statisticIndex, const T& prediction) {
            // Update the scores that are currently predicted for the example at the given index...
            scoreMatrixPtr_->addToRowFromSubset(statisticIndex, prediction.scores_cbegin(), prediction.scores_cend(),
                                                prediction.indices_cbegin(), prediction.indices_cend());

            // Update the gradients and Hessians for the example at the given index...
            lossFunctionPtr_->updateExampleWiseStatistics(statisticIndex, *labelMatrixPtr_, *scoreMatrixPtr_,
                                                          *this->statisticMatrixPtr_);
        }

    public:

        /**
         * @param lossFunctionPtr           A shared pointer to an object of type `IExampleWiseLoss`, representing the
         *                                  loss function to be used for calculating gradients and Hessians
         * @param ruleEvaluationFactoryPtr  A shared pointer to an object of type `IExampleWiseRuleEvaluationFactory`,
         *                                  to be used for calculating the predictions, as well as corresponding quality
         *                                  scores, of rules
         * @param labelMatrixPtr            A shared pointer to an object of type `IRandomAccessLabelMatrix` that
         *                                  provides random access to the labels of the training examples
         * @param statisticMatrixPtr        An unique pointer to an object of template type `StatisticMatrix` that
         *                                  stores the gradients and Hessians
         * @param scoreMatrixPtr            An unique pointer to an object of template type `ScoreMatrix` that stores
         *                                  the currently predicted scores
         */
        ExampleWiseStatistics(std::shared_ptr<IExampleWiseLoss> lossFunctionPtr,
                              std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
                              std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr,
                              std::unique_ptr<StatisticMatrix> statisticMatrixPtr,
                              std::unique_ptr<ScoreMatrix> scoreMatrixPtr)
            : ExampleWiseHistogram<StatisticVector, StatisticMatrix, ScoreMatrix>(
                  std::move(statisticMatrixPtr),
                  std::make_unique<StatisticVector>(statisticMatrixPtr->getNumCols()), ruleEvaluationFactoryPtr),
              lossFunctionPtr_(lossFunctionPtr), labelMatrixPtr_(labelMatrixPtr),
              scoreMatrixPtr_(std::move(scoreMatrixPtr)) {

        }

        void setRuleEvaluationFactory(
                std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr) override {
            this->ruleEvaluationFactoryPtr_ = ruleEvaluationFactoryPtr;
        }

        void resetSampledStatistics() override {
            // This function is equivalent to the function `resetCoveredStatistics`...
            this->resetCoveredStatistics();
        }

        void addSampledStatistic(uint32 statisticIndex, uint32 weight) override {
            // This function is equivalent to the function `updateCoveredStatistic`...
            this->updateCoveredStatistic(statisticIndex, weight, false);
        }

        void resetCoveredStatistics() override {
            this->totalSumVectorPtr_->setAllToZero();
        }

        void updateCoveredStatistic(uint32 statisticIndex, uint32 weight, bool remove) override {
            float64 signedWeight = remove ? -((float64) weight) : weight;
            this->totalSumVectorPtr_->add(this->statisticMatrixPtr_->gradients_row_cbegin(statisticIndex),
                                          this->statisticMatrixPtr_->gradients_row_cend(statisticIndex),
                                          this->statisticMatrixPtr_->hessians_row_cbegin(statisticIndex),
                                          this->statisticMatrixPtr_->hessians_row_cend(statisticIndex), signedWeight);
        }

        void applyPrediction(uint32 statisticIndex, const FullPrediction& prediction) override {
            this->applyPredictionInternally<FullPrediction>(statisticIndex, prediction);
        }

        void applyPrediction(uint32 statisticIndex, const PartialPrediction& prediction) override {
            this->applyPredictionInternally<PartialPrediction>(statisticIndex, prediction);
        }

        std::unique_ptr<IHistogramBuilder> buildHistogram(uint32 numBins) const override {
            return std::make_unique<HistogramBuilder>(*this, numBins);
        }

};

DenseExampleWiseStatisticsFactory::DenseExampleWiseStatisticsFactory(
        std::shared_ptr<IExampleWiseLoss> lossFunctionPtr,
        std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
        std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr)
    : lossFunctionPtr_(lossFunctionPtr), ruleEvaluationFactoryPtr_(ruleEvaluationFactoryPtr),
      labelMatrixPtr_(labelMatrixPtr) {

}

std::unique_ptr<IExampleWiseStatistics> DenseExampleWiseStatisticsFactory::create() const {
    uint32 numExamples = labelMatrixPtr_->getNumExamples();
    uint32 numLabels = labelMatrixPtr_->getNumLabels();
    std::unique_ptr<DenseExampleWiseStatisticMatrix> statisticMatrixPtr =
        std::make_unique<DenseExampleWiseStatisticMatrix>(numExamples, numLabels);
    std::unique_ptr<DenseNumericMatrix<float64>> scoreMatrixPtr =
        std::make_unique<DenseNumericMatrix<float64>>(numExamples, numLabels, true);

    for (uint32 r = 0; r < numExamples; r++) {
        lossFunctionPtr_->updateExampleWiseStatistics(r, *labelMatrixPtr_, *scoreMatrixPtr, *statisticMatrixPtr);
    }

    return std::make_unique<ExampleWiseStatistics<DenseExampleWiseStatisticVector, DenseExampleWiseStatisticMatrix, DenseNumericMatrix<float64>>>(
        lossFunctionPtr_, ruleEvaluationFactoryPtr_, labelMatrixPtr_, std::move(statisticMatrixPtr),
        std::move(scoreMatrixPtr));
}
