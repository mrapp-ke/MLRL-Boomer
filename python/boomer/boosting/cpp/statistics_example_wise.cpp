#include "statistics_example_wise.h"
#include "../../common/cpp/arrays.h"
#include "data.h"
#include "data_example_wise.h"
#include <cstdlib>

using namespace boosting;


/**
 * Provides access to gradients and Hessians that are calculated according to a differentiable loss function that is
 * applied example-wise using dense data structures.
 */
class DenseExampleWiseStatistics : public AbstractExampleWiseStatistics {

    private:

        /**
         * Provides access to a subset of the gradients and Hessians that are stored by an instance of the class
         * `DenseExampleWiseStatistics`.
         *
         * @tparam T The type of the vector that provides access to the indices of the labels that are included in the
         *           subset
         */
        template<class T>
        class StatisticsSubset : public IStatisticsSubset {

            private:

                const DenseExampleWiseStatistics& statistics_;

                std::unique_ptr<IExampleWiseRuleEvaluation> ruleEvaluationPtr_;

                const T& labelIndices_;

                DenseExampleWiseStatisticVector sumsOfStatistics_;

                DenseExampleWiseStatisticVector* accumulatedSumsOfStatistics_;

                const DenseExampleWiseStatisticVector* totalSumsOfStatistics_;

                DenseExampleWiseStatisticVector* totalSumsOfCoverableStatistics_;

                DenseExampleWiseStatisticVector* tmpStatistics_;

                int dsysvLwork_;

                float64* dsysvTmpArray1_;

                int* dsysvTmpArray2_;

                double* dsysvTmpArray3_;

                float64* dspmvTmpArray_;

            public:

                /**
                 * @param statistics        A reference to an object of type `DenseExampleWiseStatistics` that stores
                 *                          the gradients and Hessians
                 * @param ruleEvaluationPtr An unique pointer to an object of type `IExampleWiseRuleEvaluation` that
                 *                          should be used to calculate the predictions, as well as corresponding
                 *                          quality scores, of rules
                 * @param labelIndices      A reference to an object of template type `T` that provides access to the
                 *                          indices of the labels that are included in the subset
                 */
                StatisticsSubset(const DenseExampleWiseStatistics& statistics,
                                 std::unique_ptr<IExampleWiseRuleEvaluation> ruleEvaluationPtr, const T& labelIndices)
                    : statistics_(statistics), ruleEvaluationPtr_(std::move(ruleEvaluationPtr)),
                      labelIndices_(labelIndices),
                      sumsOfStatistics_(DenseExampleWiseStatisticVector(labelIndices.getNumElements(), true)),
                      totalSumsOfStatistics_(&statistics.totalSumsOfStatistics_) {
                    accumulatedSumsOfStatistics_ = nullptr;
                    totalSumsOfCoverableStatistics_ = nullptr;
                    tmpStatistics_ = nullptr;
                    dsysvTmpArray1_ = nullptr;
                    dsysvTmpArray2_ = nullptr;
                    dsysvTmpArray3_ = nullptr;
                    dspmvTmpArray_ = nullptr;
                }

                ~StatisticsSubset() {
                    delete accumulatedSumsOfStatistics_;
                    delete totalSumsOfCoverableStatistics_;
                    delete tmpStatistics_;
                    free(dsysvTmpArray1_);
                    free(dsysvTmpArray2_);
                    free(dsysvTmpArray3_);
                    free(dspmvTmpArray_);
                }

                void addToMissing(uint32 statisticIndex, uint32 weight) override {
                    // Create a vector for storing the totals sums of gradients and Hessians, if necessary...
                    if (totalSumsOfCoverableStatistics_ == nullptr) {
                        totalSumsOfCoverableStatistics_ = new DenseExampleWiseStatisticVector(*totalSumsOfStatistics_);
                        totalSumsOfStatistics_ = totalSumsOfCoverableStatistics_;
                    }

                    // Subtract the gradients and Hessians of the example at the given index (weighted by the given
                    // weight) from the total sums of gradients and Hessians...
                    totalSumsOfCoverableStatistics_->subtract(
                        statistics_.statistics_->gradients_row_cbegin(statisticIndex),
                        statistics_.statistics_->gradients_row_cend(statisticIndex),
                        statistics_.statistics_->hessians_row_cbegin(statisticIndex),
                        statistics_.statistics_->hessians_row_cend(statisticIndex), weight);
                }

                void addToSubset(uint32 statisticIndex, uint32 weight) override {
                    // Add the gradients and Hessians of the example at the given index (weighted by the given weight)
                    // to the current sum of gradients and Hessians...
                    uint32 numPredictions = labelIndices_.getNumElements();
                    typename T::const_iterator indexIterator = labelIndices_.cbegin();
                    uint32 i = 0;

                    DenseExampleWiseStatisticMatrix::gradient_const_iterator gradientIterator = statistics_.statistics_->gradients_row_cbegin(statisticIndex);
                    DenseExampleWiseStatisticMatrix::hessian_const_iterator hessianIterator = statistics_.statistics_->hessians_row_cbegin(statisticIndex);

                    DenseExampleWiseStatisticVector::gradient_iterator gradientSumIterator = sumsOfStatistics_.gradients_begin();
                    DenseExampleWiseStatisticVector::hessian_iterator hessianSumIterator = sumsOfStatistics_.hessians_begin();

                    for (uint32 c = 0; c < numPredictions; c++) {
                        uint32 l = indexIterator[c];
                        gradientSumIterator[c] += (weight * gradientIterator[l]);
                        uint32 offset = triangularNumber(l);

                        for (uint32 c2 = 0; c2 < c + 1; c2++) {
                            uint32 l2 = offset + indexIterator[c2];
                            hessianSumIterator[i] += (weight * hessianIterator[l2]);
                            i++;
                        }
                    }
                }

                void resetSubset() override {
                    // Create a vector for storing the accumulated sums of gradients and Hessians, if necessary...
                    if (accumulatedSumsOfStatistics_ == nullptr) {
                        uint32 numPredictions = labelIndices_.getNumElements();
                        accumulatedSumsOfStatistics_ = new DenseExampleWiseStatisticVector(numPredictions, true);
                    }

                    // Reset the sum of gradients and Hessians to zero and add it to the accumulated sums of gradients
                    // and Hessians...
                    accumulatedSumsOfStatistics_->add(sumsOfStatistics_.gradients_cbegin(),
                                                      sumsOfStatistics_.gradients_cend(),
                                                      sumsOfStatistics_.hessians_cbegin(),
                                                      sumsOfStatistics_.hessians_cend());
                    sumsOfStatistics_.setAllToZero();
                }

                const LabelWiseEvaluatedPrediction& calculateLabelWisePrediction(bool uncovered,
                                                                                 bool accumulated) override {
                    const DenseExampleWiseStatisticVector& sumsOfStatistics =
                        accumulated ? *accumulatedSumsOfStatistics_ : sumsOfStatistics_;

                    if (uncovered) {
                        uint32 numPredictions = labelIndices_.getNumElements();

                        // Initialize temporary vector, if necessary...
                        if (tmpStatistics_ == nullptr) {
                            tmpStatistics_ = new DenseExampleWiseStatisticVector(numPredictions);
                        }

                        typename T::const_iterator indexIterator = labelIndices_.cbegin();
                        DenseExampleWiseStatisticVector::gradient_iterator gradientTmpIterator = tmpStatistics_->gradients_begin();
                        DenseExampleWiseStatisticVector::gradient_const_iterator gradientTotalIterator = totalSumsOfStatistics_->gradients_cbegin();
                        DenseExampleWiseStatisticVector::hessian_iterator hessianTmpIterator = tmpStatistics_->hessians_begin();
                        DenseExampleWiseStatisticVector::hessian_const_iterator hessianTotalIterator = totalSumsOfStatistics_->hessians_cbegin();
                        DenseExampleWiseStatisticVector::gradient_const_iterator gradientSumIterator = sumsOfStatistics.gradients_cbegin();
                        DenseExampleWiseStatisticVector::hessian_const_iterator hessianSumIterator = sumsOfStatistics.hessians_cbegin();

                        for (uint32 c = 0; c < numPredictions; c++) {
                            uint32 l = indexIterator[c];
                            gradientTmpIterator[c] = gradientTotalIterator[l] - gradientSumIterator[c];
                            uint32 c2 = triangularNumber(c + 1) - 1;
                            uint32 l2 = triangularNumber(l + 1) - 1;
                            hessianTmpIterator[c2] = hessianTotalIterator[l2] - hessianSumIterator[c2];
                        }

                        return ruleEvaluationPtr_->calculateLabelWisePrediction(*tmpStatistics_);
                    }

                    return ruleEvaluationPtr_->calculateLabelWisePrediction(sumsOfStatistics);
                }

                const EvaluatedPrediction& calculateExampleWisePrediction(bool uncovered, bool accumulated) override {
                    DenseExampleWiseStatisticVector& sumsOfStatistics =
                        accumulated ? *accumulatedSumsOfStatistics_ : sumsOfStatistics_;

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

                    if (uncovered) {
                        uint32 numPredictions = labelIndices_.getNumElements();

                        // Initialize temporary vector, if necessary...
                        if (tmpStatistics_ == nullptr) {
                            tmpStatistics_ = new DenseExampleWiseStatisticVector(numPredictions);
                        }

                        typename T::const_iterator indexIterator = labelIndices_.cbegin();
                        DenseExampleWiseStatisticVector::gradient_iterator gradientTmpIterator = tmpStatistics_->gradients_begin();
                        DenseExampleWiseStatisticVector::gradient_const_iterator gradientTotalIterator = totalSumsOfStatistics_->gradients_cbegin();
                        DenseExampleWiseStatisticVector::hessian_iterator hessianTmpIterator = tmpStatistics_->hessians_begin();
                        DenseExampleWiseStatisticVector::hessian_const_iterator hessianTotalIterator = totalSumsOfStatistics_->hessians_cbegin();
                        DenseExampleWiseStatisticVector::gradient_const_iterator gradientSumIterator = sumsOfStatistics.gradients_cbegin();
                        DenseExampleWiseStatisticVector::hessian_const_iterator hessianSumIterator = sumsOfStatistics.hessians_cbegin();
                        uint32 i = 0;

                        for (uint32 c = 0; c < numPredictions; c++) {
                            uint32 l = indexIterator[c];
                            gradientTmpIterator[c] = gradientTotalIterator[l] - gradientSumIterator[c];
                            uint32 offset = triangularNumber(l);

                            for (uint32 c2 = 0; c2 < c + 1; c2++) {
                                uint32 l2 = offset + indexIterator[c2];
                                hessianTmpIterator[i] = hessianTotalIterator[l2] - hessianSumIterator[i];
                                i++;
                            }
                        }

                        return ruleEvaluationPtr_->calculateExampleWisePrediction(*tmpStatistics_, dsysvLwork_,
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
         * `DenseExampleWiseStatistics`.
         */
        class HistogramBuilder : public IHistogramBuilder {

            private:

                const DenseExampleWiseStatistics& originalStatistics_;

                DenseExampleWiseStatisticMatrix* statistics_;

            public:

            /**
             * @param statistics    A reference to an object of type `DenseExampleWiseStatistics` that stores the
             *                      gradients and Hessians
             * @param numBins       The number of bins, the histogram should consist of
             */
            HistogramBuilder(const DenseExampleWiseStatistics& statistics, uint32 numBins)
                : originalStatistics_(statistics),
                  statistics_(new DenseExampleWiseStatisticMatrix(numBins, statistics.getNumLabels(), true)) {

            }

            void onBinUpdate(uint32 binIndex, const FeatureVector::Entry& entry) override {
                uint32 index = entry.index;
                statistics_->addToRow(binIndex, originalStatistics_.statistics_->gradients_row_cbegin(index),
                                      originalStatistics_.statistics_->gradients_row_cend(index),
                                      originalStatistics_.statistics_->hessians_row_cbegin(index),
                                      originalStatistics_.statistics_->hessians_row_cend(index));
            }

            std::unique_ptr<AbstractStatistics> build() const override {
                return std::make_unique<DenseExampleWiseStatistics>(originalStatistics_.lossFunctionPtr_,
                                                                    originalStatistics_.ruleEvaluationFactoryPtr_,
                                                                    originalStatistics_.lapackPtr_,
                                                                    originalStatistics_.labelMatrixPtr_, statistics_,
                                                                    originalStatistics_.currentScores_);
            }

        };

        std::shared_ptr<IExampleWiseLoss> lossFunctionPtr_;

        std::shared_ptr<Lapack> lapackPtr_;

        std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr_;

        DenseExampleWiseStatisticMatrix* statistics_;

        DenseNumericMatrix<float64>* currentScores_;

        DenseExampleWiseStatisticVector totalSumsOfStatistics_;

        template<class T>
        void applyPredictionInternally(uint32 statisticIndex, const T& prediction) {
            // Update the scores that are currently predicted for the example at the given index...
            currentScores_->addToRowFromSubset(statisticIndex, prediction.scores_cbegin(), prediction.scores_cend(),
                                               prediction.indices_cbegin(), prediction.indices_cend());

            // Update the gradients and Hessians for the example at the given index...
            lossFunctionPtr_->updateStatistics(statisticIndex, *labelMatrixPtr_, *currentScores_, *statistics_);
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
         * @param gradients                 A pointer to an array of type `float64`, shape `(num_examples, num_labels)`,
         *                                  representing the gradients
         * @param hessians                  A pointer to an array of type `float64`, shape
         *                                  `(num_examples, num_labels + (num_labels + 1) // 2)`, representing the
         *                                  Hessians
         * @param currentScores             A pointer to an array of type `float64`, shape `(num_examples, num_labels`),
         *                                  representing the currently predicted scores
         */
        DenseExampleWiseStatistics(std::shared_ptr<IExampleWiseLoss> lossFunctionPtr,
                                   std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
                                   std::shared_ptr<Lapack> lapackPtr,
                                   std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr,
                                   DenseExampleWiseStatisticMatrix* statistics,
                                   DenseNumericMatrix<float64>* currentScores)
            : AbstractExampleWiseStatistics(labelMatrixPtr->getNumExamples(), labelMatrixPtr->getNumLabels(),
                                            ruleEvaluationFactoryPtr),
              lossFunctionPtr_(lossFunctionPtr), lapackPtr_(lapackPtr), labelMatrixPtr_(labelMatrixPtr),
              statistics_(statistics), currentScores_(currentScores),
              totalSumsOfStatistics_(DenseExampleWiseStatisticVector(labelMatrixPtr->getNumLabels())) {

        }

        ~DenseExampleWiseStatistics() {
            delete statistics_;
            delete currentScores_;
        }

        void resetCoveredStatistics() override {
            totalSumsOfStatistics_.setAllToZero();
        }

        void updateCoveredStatistic(uint32 statisticIndex, uint32 weight, bool remove) override {
            float64 signedWeight = remove ? -((float64) weight) : weight;
            totalSumsOfStatistics_.add(statistics_->gradients_row_cbegin(statisticIndex),
                                       statistics_->gradients_row_cend(statisticIndex),
                                       statistics_->hessians_row_cbegin(statisticIndex),
                                       statistics_->hessians_row_cend(statisticIndex), signedWeight);
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
    : AbstractGradientStatistics(numStatistics, numLabels), ruleEvaluationFactoryPtr_(ruleEvaluationFactoryPtr) {

}

void AbstractExampleWiseStatistics::setRuleEvaluationFactory(
        std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr) {
    ruleEvaluationFactoryPtr_ = ruleEvaluationFactoryPtr;
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
    DenseExampleWiseStatisticMatrix* statistics = new DenseExampleWiseStatisticMatrix(numExamples, numLabels);
    DenseNumericMatrix<float64>* currentScores = new DenseNumericMatrix<float64>(numExamples, numLabels, true);

    for (uint32 r = 0; r < numExamples; r++) {
        lossFunctionPtr_->updateStatistics(r, *labelMatrixPtr_, *currentScores, *statistics);
    }

    return std::make_unique<DenseExampleWiseStatistics>(lossFunctionPtr_, ruleEvaluationFactoryPtr_, lapackPtr_,
                                                        labelMatrixPtr_, statistics, currentScores);
}
