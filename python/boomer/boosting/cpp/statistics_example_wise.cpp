#include "statistics_example_wise.h"
#include "data.h"
#include "data_example_wise.h"
#include <cstdlib>

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
class ExampleWiseStatistics : public AbstractExampleWiseStatistics {

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

                const ExampleWiseStatistics& statistics_;

                std::unique_ptr<IExampleWiseRuleEvaluation> ruleEvaluationPtr_;

                const T& labelIndices_;

                StatisticVector sumVector_;

                StatisticVector* accumulatedSumVector_;

                const StatisticVector* totalSumVector_;

                StatisticVector* totalCoverableSumVector_;

                StatisticVector* tmpVector_;

                int dsysvLwork_;

                float64* dsysvTmpArray1_;

                int* dsysvTmpArray2_;

                double* dsysvTmpArray3_;

                float64* dspmvTmpArray_;

            public:

                /**
                 * @param statistics        A reference to an object of type `ExampleWiseStatistics` that stores the
                 *                          gradients and Hessians
                 * @param ruleEvaluationPtr An unique pointer to an object of type `IExampleWiseRuleEvaluation` that
                 *                          should be used to calculate the predictions, as well as corresponding
                 *                          quality scores, of rules
                 * @param labelIndices      A reference to an object of template type `T` that provides access to the
                 *                          indices of the labels that are included in the subset
                 */
                StatisticsSubset(const ExampleWiseStatistics& statistics,
                                 std::unique_ptr<IExampleWiseRuleEvaluation> ruleEvaluationPtr, const T& labelIndices)
                    : statistics_(statistics), ruleEvaluationPtr_(std::move(ruleEvaluationPtr)),
                      labelIndices_(labelIndices), sumVector_(StatisticVector(labelIndices.getNumElements(), true)),
                      totalSumVector_(&statistics.totalSumVector_) {
                    accumulatedSumVector_ = nullptr;
                    totalCoverableSumVector_ = nullptr;
                    tmpVector_ = nullptr;
                    dsysvTmpArray1_ = nullptr;
                    dsysvTmpArray2_ = nullptr;
                    dsysvTmpArray3_ = nullptr;
                    dspmvTmpArray_ = nullptr;
                }

                ~StatisticsSubset() {
                    delete accumulatedSumVector_;
                    delete totalCoverableSumVector_;
                    delete tmpVector_;
                    free(dsysvTmpArray1_);
                    free(dsysvTmpArray2_);
                    free(dsysvTmpArray3_);
                    free(dspmvTmpArray_);
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
                        statistics_.statisticMatrixPtr_->gradients_row_cbegin(statisticIndex),
                        statistics_.statisticMatrixPtr_->gradients_row_cend(statisticIndex),
                        statistics_.statisticMatrixPtr_->hessians_row_cbegin(statisticIndex),
                        statistics_.statisticMatrixPtr_->hessians_row_cend(statisticIndex), weight);
                }

                void addToSubset(uint32 statisticIndex, uint32 weight) override {
                    sumVector_.addToSubset(statistics_.statisticMatrixPtr_->gradients_row_cbegin(statisticIndex),
                                           statistics_.statisticMatrixPtr_->gradients_row_cend(statisticIndex),
                                           statistics_.statisticMatrixPtr_->hessians_row_cbegin(statisticIndex),
                                           statistics_.statisticMatrixPtr_->hessians_row_cend(statisticIndex),
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

                const LabelWiseEvaluatedPrediction& calculateLabelWisePrediction(bool uncovered,
                                                                                 bool accumulated) override {
                    const StatisticVector& sumsOfStatistics = accumulated ? *accumulatedSumVector_ : sumVector_;

                    if (uncovered) {

                        // Initialize temporary vector, if necessary...
                        if (tmpVector_ == nullptr) {
                            uint32 numPredictions = labelIndices_.getNumElements();
                            tmpVector_ = new StatisticVector(numPredictions);
                        }

                        tmpVector_->difference(totalSumVector_->gradients_cbegin(), totalSumVector_->gradients_cend(),
                                               totalSumVector_->hessians_cbegin(), totalSumVector_->hessians_cend(),
                                               labelIndices_, sumsOfStatistics.gradients_cbegin(),
                                               sumsOfStatistics.gradients_cend(), sumsOfStatistics.hessians_cbegin(),
                                               sumsOfStatistics.hessians_cend());
                        return ruleEvaluationPtr_->calculateLabelWisePrediction(*tmpVector_);
                    }

                    return ruleEvaluationPtr_->calculateLabelWisePrediction(sumsOfStatistics);
                }

                const EvaluatedPrediction& calculateExampleWisePrediction(bool uncovered, bool accumulated) override {
                    // To avoid array recreation each time this function is called, the temporary arrays are only
                    // initialized if they have not been initialized yet
                    if (dsysvTmpArray1_ == nullptr) {
                        uint32 numPredictions = labelIndices_.getNumElements();
                        dsysvTmpArray1_ = (float64*) malloc(numPredictions * numPredictions * sizeof(float64));
                        dsysvTmpArray2_ = (int*) malloc(numPredictions * sizeof(int));
                        dspmvTmpArray_ = (float64*) malloc(numPredictions * sizeof(float64));

                        // Query the optimal "lwork" parameter to be used by LAPACK's DSYSV routine...
                        dsysvLwork_ = statistics_.lapackPtr_->queryDsysvLworkParameter(dsysvTmpArray1_, dspmvTmpArray_,
                                                                                       numPredictions);
                        dsysvTmpArray3_ = (double*) malloc(dsysvLwork_ * sizeof(double));
                    }

                    StatisticVector& sumsOfStatistics = accumulated ? *accumulatedSumVector_ : sumVector_;

                    if (uncovered) {
                        // Initialize temporary vector, if necessary...
                        if (tmpVector_ == nullptr) {
                            uint32 numPredictions = labelIndices_.getNumElements();
                            tmpVector_ = new StatisticVector(numPredictions);
                        }

                        tmpVector_->difference(totalSumVector_->gradients_cbegin(), totalSumVector_->gradients_cend(),
                                               totalSumVector_->hessians_cbegin(), totalSumVector_->hessians_cend(),
                                               labelIndices_, sumsOfStatistics.gradients_cbegin(),
                                               sumsOfStatistics.gradients_cend(), sumsOfStatistics.hessians_cbegin(),
                                               sumsOfStatistics.hessians_cend());
                        return ruleEvaluationPtr_->calculateExampleWisePrediction(*tmpVector_, dsysvLwork_,
                                                                                  dsysvTmpArray1_, dsysvTmpArray2_,
                                                                                  dsysvTmpArray3_, dspmvTmpArray_);
                    }

                    return ruleEvaluationPtr_->calculateExampleWisePrediction(sumsOfStatistics, dsysvLwork_,
                                                                              dsysvTmpArray1_, dsysvTmpArray2_,
                                                                              dsysvTmpArray3_, dspmvTmpArray_);
                }

        };

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

            void onBinUpdate(uint32 binIndex, const FeatureVector::Entry& entry) override {
                uint32 index = entry.index;
                statisticMatrixPtr_->addToRow(binIndex, statistics_.statisticMatrixPtr_->gradients_row_cbegin(index),
                                              statistics_.statisticMatrixPtr_->gradients_row_cend(index),
                                              statistics_.statisticMatrixPtr_->hessians_row_cbegin(index),
                                              statistics_.statisticMatrixPtr_->hessians_row_cend(index));
            }

            std::unique_ptr<IHistogram> build() override {
                return std::make_unique<ExampleWiseStatistics<StatisticVector, StatisticMatrix, ScoreMatrix>>(
                    statistics_.lossFunctionPtr_, statistics_.ruleEvaluationFactoryPtr_, statistics_.lapackPtr_,
                    statistics_.labelMatrixPtr_, std::move(statisticMatrixPtr_), nullptr);
            }

        };

        std::shared_ptr<IExampleWiseLoss> lossFunctionPtr_;

        std::shared_ptr<Lapack> lapackPtr_;

        std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr_;

        std::unique_ptr<StatisticMatrix> statisticMatrixPtr_;

        std::unique_ptr<ScoreMatrix> scoreMatrixPtr_;

        StatisticVector totalSumVector_;

        template<class T>
        void applyPredictionInternally(uint32 statisticIndex, const T& prediction) {
            // Update the scores that are currently predicted for the example at the given index...
            scoreMatrixPtr_->addToRowFromSubset(statisticIndex, prediction.scores_cbegin(), prediction.scores_cend(),
                                                prediction.indices_cbegin(), prediction.indices_cend());

            // Update the gradients and Hessians for the example at the given index...
            lossFunctionPtr_->updateExampleWiseStatistics(statisticIndex, *labelMatrixPtr_, *scoreMatrixPtr_,
                                                          *statisticMatrixPtr_);
        }

    public:

        /**
         * @param lossFunctionPtr           A shared pointer to an object of type `IExampleWiseLoss`, representing the
         *                                  loss function to be used for calculating gradients and Hessians
         * @param ruleEvaluationFactoryPtr  A shared pointer to an object of type `IExampleWiseRuleEvaluationFactory`,
         *                                  to be used for calculating the predictions, as well as corresponding quality
         *                                  scores, of rules
         * @param lapackPtr                 A shared pointer to an object of type `Lapack` that allows to execute
         *                                  different Lapack routines
         * @param labelMatrixPtr            A shared pointer to an object of type `IRandomAccessLabelMatrix` that
         *                                  provides random access to the labels of the training examples
         * @param statisticMatrixPtr        An unique pointer to an object of template type `StatisticMatrix` that
         *                                  stores the gradients and Hessians
         * @param scoreMatrixPtr            An unique pointer to an object of template type `ScoreMatrix` that stores
         *                                  the currently predicted scores
         */
        ExampleWiseStatistics(std::shared_ptr<IExampleWiseLoss> lossFunctionPtr,
                              std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
                              std::shared_ptr<Lapack> lapackPtr,
                              std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr,
                              std::unique_ptr<StatisticMatrix> statisticMatrixPtr,
                              std::unique_ptr<ScoreMatrix> scoreMatrixPtr)
            : AbstractExampleWiseStatistics(labelMatrixPtr->getNumExamples(), labelMatrixPtr->getNumLabels(),
                                            ruleEvaluationFactoryPtr),
              lossFunctionPtr_(lossFunctionPtr), lapackPtr_(lapackPtr), labelMatrixPtr_(labelMatrixPtr),
              statisticMatrixPtr_(std::move(statisticMatrixPtr)), scoreMatrixPtr_(std::move(scoreMatrixPtr)),
              totalSumVector_(StatisticVector(labelMatrixPtr->getNumLabels())) {

        }

        void resetCoveredStatistics() override {
            totalSumVector_.setAllToZero();
        }

        void updateCoveredStatistic(uint32 statisticIndex, uint32 weight, bool remove) override {
            float64 signedWeight = remove ? -((float64) weight) : weight;
            totalSumVector_.add(statisticMatrixPtr_->gradients_row_cbegin(statisticIndex),
                                statisticMatrixPtr_->gradients_row_cend(statisticIndex),
                                statisticMatrixPtr_->hessians_row_cbegin(statisticIndex),
                                statisticMatrixPtr_->hessians_row_cend(statisticIndex), signedWeight);
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

AbstractExampleWiseStatistics::AbstractExampleWiseStatistics(
        uint32 numStatistics, uint32 numLabels,
        std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr)
    : numStatistics_(numStatistics), numLabels_(numLabels), ruleEvaluationFactoryPtr_(ruleEvaluationFactoryPtr) {

}

void AbstractExampleWiseStatistics::setRuleEvaluationFactory(
        std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr) {
    ruleEvaluationFactoryPtr_ = ruleEvaluationFactoryPtr;
}

uint32 AbstractExampleWiseStatistics::getNumStatistics() const {
    return numStatistics_;
}

uint32 AbstractExampleWiseStatistics::getNumLabels() const {
    return numLabels_;
}

void AbstractExampleWiseStatistics::resetSampledStatistics() {
    // This function is equivalent to the function `resetCoveredStatistics`...
    this->resetCoveredStatistics();
}

void AbstractExampleWiseStatistics::addSampledStatistic(uint32 statisticIndex, uint32 weight) {
    // This function is equivalent to the function `updateCoveredStatistic`...
    this->updateCoveredStatistic(statisticIndex, weight, false);
}

DenseExampleWiseStatisticsFactoryImpl::DenseExampleWiseStatisticsFactoryImpl(
        std::shared_ptr<IExampleWiseLoss> lossFunctionPtr,
        std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr, std::unique_ptr<Lapack> lapackPtr,
        std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr) {
    lossFunctionPtr_ = lossFunctionPtr;
    ruleEvaluationFactoryPtr_ = ruleEvaluationFactoryPtr;
    lapackPtr_ = std::move(lapackPtr);
    labelMatrixPtr_ = labelMatrixPtr;
}

std::unique_ptr<AbstractExampleWiseStatistics> DenseExampleWiseStatisticsFactoryImpl::create() const {
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
        lossFunctionPtr_, ruleEvaluationFactoryPtr_, lapackPtr_, labelMatrixPtr_, std::move(statisticMatrixPtr),
        std::move(scoreMatrixPtr));
}
