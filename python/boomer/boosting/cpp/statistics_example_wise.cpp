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

                DenseExampleWiseStatisticsVector sumsOfStatistics_;

                DenseExampleWiseStatisticsVector* accumulatedSumsOfStatistics_;

                const DenseExampleWiseStatisticsVector* totalSumsOfStatistics_;

                DenseExampleWiseStatisticsVector* totalSumsOfCoverableStatistics_;

                DenseExampleWiseStatisticsVector* tmpStatistics_;

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
                      sumsOfStatistics_(DenseExampleWiseStatisticsVector(labelIndices.getNumElements(), true)),
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
                    uint32 numLabels = statistics_.getNumLabels();
                    uint32 numHessians = triangularNumber(numLabels);

                    // Allocate arrays for storing the totals sums of gradients and Hessians, if necessary...
                    if (totalSumsOfCoverableStatistics_ == nullptr) {
                        totalSumsOfCoverableStatistics_ = new DenseExampleWiseStatisticsVector(*totalSumsOfStatistics_);
                        totalSumsOfStatistics_ = totalSumsOfCoverableStatistics_;
                    }

                    // For each label, subtract the gradient and Hessian of the example at the given index (weighted by
                    // the given weight) from the total sum of gradients and Hessians...
                    DenseExampleWiseStatisticsMatrix::gradient_const_iterator gradientIterator = statistics_.statistics_->gradients_row_cbegin(statisticIndex);
                    DenseExampleWiseStatisticsVector::gradient_iterator gradientSumIterator = totalSumsOfCoverableStatistics_->gradients_begin();

                    for (uint32 c = 0; c < numLabels; c++) {
                        gradientSumIterator[c] -= (weight * gradientIterator[c]);
                    }

                    DenseExampleWiseStatisticsMatrix::hessian_const_iterator hessianIterator = statistics_.statistics_->hessians_row_cbegin(statisticIndex);
                    DenseExampleWiseStatisticsVector::hessian_iterator hessianSumIterator = totalSumsOfCoverableStatistics_->hessians_begin();

                    for (uint32 c = 0; c < numHessians; c++) {
                        hessianSumIterator[c] -= (weight * hessianIterator[c]);
                    }
                }

                void addToSubset(uint32 statisticIndex, uint32 weight) override {
                    // Add the gradients and Hessians of the example at the given index (weighted by the given weight)
                    // to the current sum of gradients and Hessians...
                    uint32 numPredictions = labelIndices_.getNumElements();
                    typename T::const_iterator indexIterator = labelIndices_.cbegin();
                    uint32 i = 0;

                    DenseExampleWiseStatisticsMatrix::gradient_const_iterator gradientIterator = statistics_.statistics_->gradients_row_cbegin(statisticIndex);
                    DenseExampleWiseStatisticsMatrix::hessian_const_iterator hessianIterator = statistics_.statistics_->hessians_row_cbegin(statisticIndex);

                    DenseExampleWiseStatisticsVector::gradient_iterator gradientSumIterator = sumsOfStatistics_.gradients_begin();
                    DenseExampleWiseStatisticsVector::hessian_iterator hessianSumIterator = sumsOfStatistics_.hessians_begin();

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
                    uint32 numPredictions = labelIndices_.getNumElements();
                    uint32 numHessians = triangularNumber(numPredictions);

                    // Allocate arrays for storing the accumulated sums of gradients and Hessians, if necessary...
                    if (accumulatedSumsOfStatistics_ == nullptr) {
                        accumulatedSumsOfStatistics_ = new DenseExampleWiseStatisticsVector(numPredictions, true);
                    }

                    // Reset the sum of gradients and Hessians for each label to zero and add it to the accumulated sums
                    // of gradients and Hessians...
                    DenseExampleWiseStatisticsVector::gradient_const_iterator gradientIterator = sumsOfStatistics_.gradients_cbegin();
                    DenseExampleWiseStatisticsVector::gradient_iterator gradientAccIterator = accumulatedSumsOfStatistics_->gradients_begin();

                    for (uint32 c = 0; c < numPredictions; c++) {
                        gradientAccIterator[c] += gradientIterator[c];
                    }

                    DenseExampleWiseStatisticsVector::hessian_const_iterator hessianIterator = sumsOfStatistics_.hessians_cbegin();
                    DenseExampleWiseStatisticsVector::hessian_iterator hessianAccIterator = accumulatedSumsOfStatistics_->hessians_begin();

                    for (uint32 c = 0; c < numHessians; c++) {
                        hessianAccIterator[c] += hessianIterator[c];
                    }

                    sumsOfStatistics_.setAllToZero();
                }

                const LabelWiseEvaluatedPrediction& calculateLabelWisePrediction(bool uncovered,
                                                                                 bool accumulated) override {
                    const DenseExampleWiseStatisticsVector& sumsOfStatistics =
                        accumulated ? *accumulatedSumsOfStatistics_ : sumsOfStatistics_;

                    if (uncovered) {
                        uint32 numPredictions = labelIndices_.getNumElements();

                        // Initialize temporary vector, if necessary...
                        if (tmpStatistics_ == nullptr) {
                            tmpStatistics_ = new DenseExampleWiseStatisticsVector(numPredictions);
                        }

                        typename T::const_iterator indexIterator = labelIndices_.cbegin();
                        DenseExampleWiseStatisticsVector::gradient_iterator gradientTmpIterator = tmpStatistics_->gradients_begin();
                        DenseExampleWiseStatisticsVector::gradient_const_iterator gradientTotalIterator = totalSumsOfStatistics_->gradients_cbegin();
                        DenseExampleWiseStatisticsVector::hessian_iterator hessianTmpIterator = tmpStatistics_->hessians_begin();
                        DenseExampleWiseStatisticsVector::hessian_const_iterator hessianTotalIterator = totalSumsOfStatistics_->hessians_cbegin();
                        DenseExampleWiseStatisticsVector::gradient_const_iterator gradientSumIterator = sumsOfStatistics.gradients_cbegin();
                        DenseExampleWiseStatisticsVector::hessian_const_iterator hessianSumIterator = sumsOfStatistics.hessians_cbegin();

                        for (uint32 c = 0; c < numPredictions; c++) {
                            uint32 l = indexIterator[c];
                            gradientTmpIterator[c] = gradientTotalIterator[l] - gradientSumIterator[c];
                            uint32 c2 = triangularNumber(c + 1) - 1;
                            uint32 l2 = triangularNumber(l + 1) - 1;
                            hessianTmpIterator[c2] = hessianTotalIterator[l2] - hessianSumIterator[c2];
                        }

                        return ruleEvaluationPtr_->calculateLabelWisePrediction(tmpStatistics_->gradients_begin(),
                                                                                tmpStatistics_->hessians_begin());
                    }

                    return ruleEvaluationPtr_->calculateLabelWisePrediction(sumsOfStatistics.gradients_cbegin(),
                                                                            sumsOfStatistics.hessians_cbegin());
                }

                const EvaluatedPrediction& calculateExampleWisePrediction(bool uncovered, bool accumulated) override {
                    DenseExampleWiseStatisticsVector& sumsOfStatistics =
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
                            tmpStatistics_ = new DenseExampleWiseStatisticsVector(numPredictions);
                        }

                        typename T::const_iterator indexIterator = labelIndices_.cbegin();
                        DenseExampleWiseStatisticsVector::gradient_iterator gradientTmpIterator = tmpStatistics_->gradients_begin();
                        DenseExampleWiseStatisticsVector::gradient_const_iterator gradientTotalIterator = totalSumsOfStatistics_->gradients_cbegin();
                        DenseExampleWiseStatisticsVector::hessian_iterator hessianTmpIterator = tmpStatistics_->hessians_begin();
                        DenseExampleWiseStatisticsVector::hessian_const_iterator hessianTotalIterator = totalSumsOfStatistics_->hessians_cbegin();
                        DenseExampleWiseStatisticsVector::gradient_const_iterator gradientSumIterator = sumsOfStatistics.gradients_cbegin();
                        DenseExampleWiseStatisticsVector::hessian_const_iterator hessianSumIterator = sumsOfStatistics.hessians_cbegin();
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

                        return ruleEvaluationPtr_->calculateExampleWisePrediction(tmpStatistics_->gradients_begin(),
                                                                                  tmpStatistics_->hessians_begin(),
                                                                                  dsysvLwork_, dsysvTmpArray1_,
                                                                                  dsysvTmpArray2_, dsysvTmpArray3_,
                                                                                  dspmvTmpArray_);
                    }

                    return ruleEvaluationPtr_->calculateExampleWisePrediction(sumsOfStatistics.gradients_begin(),
                                                                              sumsOfStatistics.hessians_begin(),
                                                                              dsysvLwork_, dsysvTmpArray1_,
                                                                              dsysvTmpArray2_, dsysvTmpArray3_,
                                                                              dspmvTmpArray_);
                }

        };

        /**
         * Allows to build a histogram based on the gradients and Hessians that are stored by an instance of the class
         * `DenseExampleWiseStatistics`.
         */
        class HistogramBuilder : public IHistogramBuilder {

            private:

                const DenseExampleWiseStatistics& originalStatistics_;

                DenseExampleWiseStatisticsMatrix* statistics_;

            public:

            /**
             * @param statistics    A reference to an object of type `DenseExampleWiseStatistics` that stores the
             *                      gradients and Hessians
             * @param numBins       The number of bins, the histogram should consist of
             */
            HistogramBuilder(const DenseExampleWiseStatistics& statistics, uint32 numBins)
                : originalStatistics_(statistics),
                  statistics_(new DenseExampleWiseStatisticsMatrix(numBins, statistics.getNumLabels(), true)) {

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

        DenseExampleWiseStatisticsMatrix* statistics_;

        DenseNumericMatrix<float64>* currentScores_;

        DenseExampleWiseStatisticsVector totalSumsOfStatistics_;

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
                                   DenseExampleWiseStatisticsMatrix* statistics,
                                   DenseNumericMatrix<float64>* currentScores)
            : AbstractExampleWiseStatistics(labelMatrixPtr->getNumExamples(), labelMatrixPtr->getNumLabels(),
                                            ruleEvaluationFactoryPtr),
              lossFunctionPtr_(lossFunctionPtr), lapackPtr_(lapackPtr), labelMatrixPtr_(labelMatrixPtr),
              statistics_(statistics), currentScores_(currentScores),
              totalSumsOfStatistics_(DenseExampleWiseStatisticsVector(labelMatrixPtr->getNumLabels())) {

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
    DenseExampleWiseStatisticsMatrix* statistics = new DenseExampleWiseStatisticsMatrix(numExamples, numLabels);
    DenseNumericMatrix<float64>* currentScores = new DenseNumericMatrix<float64>(numExamples, numLabels, true);

    for (uint32 r = 0; r < numExamples; r++) {
        lossFunctionPtr_->updateStatistics(r, *labelMatrixPtr_, *currentScores, *statistics);
    }

    return std::make_unique<DenseExampleWiseStatistics>(lossFunctionPtr_, ruleEvaluationFactoryPtr_, lapackPtr_,
                                                        labelMatrixPtr_, statistics, currentScores);
}
