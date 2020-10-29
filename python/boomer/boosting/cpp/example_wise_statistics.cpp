#include "example_wise_statistics.h"
#include "linalg.h"
#include <cstdlib>

using namespace boosting;


/**
 * Provides access to a subset of the gradients and Hessians that are stored by an instance of the class
 * `DenseExampleWiseStatisticsImpl`.
 *
 * @tparam T The type of the vector that provides access to the indices of the labels that are included in the subset
 */
template<class T>
class DenseExampleWiseStatisticsImpl::StatisticsSubset : virtual public IStatisticsSubset {

    private:

        const DenseExampleWiseStatisticsImpl& statistics_;

        std::unique_ptr<IExampleWiseRuleEvaluation> ruleEvaluationPtr_;

        const T& labelIndices_;

        float64* sumsOfGradients_;

        float64* accumulatedSumsOfGradients_;

        float64* sumsOfHessians_;

        float64* accumulatedSumsOfHessians_;

        float64* tmpGradients_;

        float64* tmpHessians_;

        int dsysvLwork_;

        float64* dsysvTmpArray1_;

        int* dsysvTmpArray2_;

        double* dsysvTmpArray3_;

        float64* dspmvTmpArray_;

    public:

        /**
         * @param statistics        A reference to an object of type `DenseExampleWiseStatisticsImpl` that stores the
         *                          gradients and Hessians
         * @param ruleEvaluationPtr An unique pointer to an object of type `IExampleWiseRuleEvaluation` that should be
         *                          used to calculate the predictions, as well as corresponding quality scores, of rules
         * @param labelIndices      A reference to an object of template type `T` that provides access to the indices of
         *                          the labels that are included in the subset
         */
        StatisticsSubset(const DenseExampleWiseStatisticsImpl& statistics,
                         std::unique_ptr<IExampleWiseRuleEvaluation> ruleEvaluationPtr, const T& labelIndices)
            : statistics_(statistics), ruleEvaluationPtr_(std::move(ruleEvaluationPtr)), labelIndices_(labelIndices) {
            uint32 numPredictions = labelIndices.getNumElements();
            sumsOfGradients_ = (float64*) malloc(numPredictions * sizeof(float64));
            arrays::setToZeros(sumsOfGradients_, numPredictions);
            accumulatedSumsOfGradients_ = nullptr;
            uint32 numHessians = linalg::triangularNumber(numPredictions);
            sumsOfHessians_ = (float64*) malloc(numHessians * sizeof(float64));
            arrays::setToZeros(sumsOfHessians_, numHessians);
            accumulatedSumsOfHessians_ = nullptr;
            tmpGradients_ = nullptr;
            tmpHessians_ = nullptr;
            dsysvTmpArray1_ = nullptr;
            dsysvTmpArray2_ = nullptr;
            dsysvTmpArray3_ = nullptr;
            dspmvTmpArray_ = nullptr;
        }

        ~StatisticsSubset() {
            free(sumsOfGradients_);
            free(accumulatedSumsOfGradients_);
            free(sumsOfHessians_);
            free(accumulatedSumsOfHessians_);
            free(tmpGradients_);
            free(tmpHessians_);
            free(dsysvTmpArray1_);
            free(dsysvTmpArray2_);
            free(dsysvTmpArray3_);
            free(dspmvTmpArray_);
        }

        void addToSubset(uint32 statisticIndex, uint32 weight) override {
            // Add the gradients and Hessians of the example at the given index (weighted by the given weight) to the
            // current sum of gradients and Hessians...
            uint32 numLabels = statistics_.getNumCols();
            uint32 offsetGradients = statisticIndex * numLabels;
            uint32 offsetHessians = statisticIndex * linalg::triangularNumber(numLabels);
            uint32 numPredictions = labelIndices_.getNumElements();
            typename T::index_const_iterator indexIterator = labelIndices_.indices_cbegin();
            uint32 i = 0;

            for (uint32 c = 0; c < numPredictions; c++) {
                uint32 l = indexIterator[c];
                sumsOfGradients_[c] += (weight * statistics_.gradients_[offsetGradients + l]);
                uint32 triangularNumber = linalg::triangularNumber(l);

                for (uint32 c2 = 0; c2 < c + 1; c2++) {
                    uint32 l2 = triangularNumber + indexIterator[c2];
                    sumsOfHessians_[i] += (weight * statistics_.hessians_[offsetHessians + l2]);
                    i++;
                }
            }
        }

        void resetSubset() override {
            uint32 numPredictions = labelIndices_.getNumElements();
            uint32 numHessians = linalg::triangularNumber(numPredictions);

            // Allocate arrays for storing the accumulated sums of gradients and Hessians, if necessary...
            if (accumulatedSumsOfGradients_ == nullptr) {
                accumulatedSumsOfGradients_ = (float64*) malloc(numPredictions * sizeof(float64));
                arrays::setToZeros(accumulatedSumsOfGradients_, numPredictions);
                accumulatedSumsOfHessians_ = (float64*) malloc(numHessians * sizeof(float64));
                arrays::setToZeros(accumulatedSumsOfHessians_, numHessians);
            }

            // Reset the sum of gradients and Hessians for each label to zero and add it to the accumulated sums of
            // gradients and Hessians...
            for (uint32 c = 0; c < numPredictions; c++) {
                accumulatedSumsOfGradients_[c] += sumsOfGradients_[c];
                sumsOfGradients_[c] = 0;
            }

            for (uint32 c = 0; c < numHessians; c++) {
                accumulatedSumsOfHessians_[c] += sumsOfHessians_[c];
                sumsOfHessians_[c] = 0;
            }
        }

        const LabelWiseEvaluatedPrediction& calculateLabelWisePrediction(bool uncovered, bool accumulated) override {
            float64* sumsOfGradients = accumulated ? accumulatedSumsOfGradients_ : sumsOfGradients_;
            float64* sumsOfHessians = accumulated ? accumulatedSumsOfHessians_ : sumsOfHessians_;
            return ruleEvaluationPtr_->calculateLabelWisePrediction(statistics_.totalSumsOfGradients_, sumsOfGradients,
                                                                    statistics_.totalSumsOfHessians_, sumsOfHessians,
                                                                    uncovered);
        }

        const EvaluatedPrediction& calculateExampleWisePrediction(bool uncovered, bool accumulated) override {
            uint32 numPredictions = labelIndices_.getNumElements();
            float64* sumsOfGradients = accumulated ? accumulatedSumsOfGradients_ : sumsOfGradients_;
            float64* sumsOfHessians = accumulated ? accumulatedSumsOfHessians_ : sumsOfHessians_;

            // To avoid array recreation each time this function is called, the temporary arrays are only initialized if they
            // have not been initialized yet
            if (tmpGradients_ == nullptr) {
                tmpGradients_ = (float64*) malloc(numPredictions * sizeof(float64));
                uint32 numHessians = linalg::triangularNumber(numPredictions);
                tmpHessians_ = (float64*) malloc(numHessians * sizeof(float64));
                dsysvTmpArray1_ = (float64*) malloc(numPredictions * numPredictions * sizeof(float64));
                dsysvTmpArray2_ = (int*) malloc(numPredictions * sizeof(int));
                dspmvTmpArray_ = (float64*) malloc(numPredictions * sizeof(float64));

                // Query the optimal "lwork" parameter to be used by LAPACK'S DSYSV routine...
                dsysvLwork_ = statistics_.lapackPtr_->queryDsysvLworkParameter(dsysvTmpArray1_, tmpGradients_,
                                                                               numPredictions);
                dsysvTmpArray3_ = (double*) malloc(dsysvLwork_ * sizeof(double));
            }

            return ruleEvaluationPtr_->calculateExampleWisePrediction(statistics_.totalSumsOfGradients_,
                                                                      sumsOfGradients, statistics_.totalSumsOfHessians_,
                                                                      sumsOfHessians, tmpGradients_, tmpHessians_,
                                                                      dsysvLwork_, dsysvTmpArray1_, dsysvTmpArray2_,
                                                                      dsysvTmpArray3_, dspmvTmpArray_, uncovered);
        }

};

/**
 * Allows to build a histogram based on the gradients and Hessians that are stored by an instance of the class
 * `DenseExampleWiseStatisticsImpl`.
 */
class DenseExampleWiseStatisticsImpl::HistogramBuilder : virtual public IHistogramBuilder {

    private:

        const DenseExampleWiseStatisticsImpl& statistics_;

        uint32 numBins_;

        float64* gradients_;

        float64* hessians_;

    public:

    /**
     * @param statistics    A reference to an object of type `DenseExampleWiseStatisticsImpl` that stores the gradients
     *                      and Hessians
     * @param numBins       The number of bins, the histogram should consist of
     */
    HistogramBuilder(const DenseExampleWiseStatisticsImpl& statistics, uint32 numBins)
        : statistics_(statistics), numBins_(numBins) {
        uint32 numGradients = statistics.getNumCols();
        uint32 numHessians = linalg::triangularNumber(numGradients);
        gradients_ = (float64*) calloc((numBins_ * numGradients), sizeof(float64));
        hessians_ = (float64*) calloc((numBins_ * numHessians), sizeof(float64));
    }

    void onBinUpdate(uint32 binIndex, const FeatureVector::Entry& entry) override {
        uint32 numLabels = statistics_.getNumCols();
        uint32 index = entry.index;
        uint32 offset = index * numLabels;
        uint32 gradientOffset = binIndex * numLabels;
        uint32 hessianOffset = binIndex * linalg::triangularNumber(numLabels);

        for(uint32 c = 0; c < numLabels; c++) {
            float64 gradient = statistics_.gradients_[offset + c];
            float64 hessian = statistics_.hessians_[offset + c];
            gradients_[gradientOffset + c] += gradient;
            hessians_[hessianOffset + c] += hessian;
        }
    }

    std::unique_ptr<AbstractStatistics> build() const override {
        return std::make_unique<DenseExampleWiseStatisticsImpl>(statistics_.lossFunctionPtr_,
                                                                statistics_.ruleEvaluationFactoryPtr_,
                                                                statistics_.lapackPtr_, statistics_.labelMatrixPtr_,
                                                                gradients_, hessians_, statistics_.currentScores_);
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

DenseExampleWiseStatisticsImpl::DenseExampleWiseStatisticsImpl(
        std::shared_ptr<IExampleWiseLoss> lossFunctionPtr,
        std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr, std::shared_ptr<Lapack> lapackPtr,
        std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr, float64* gradients, float64* hessians,
        float64* currentScores)
    : AbstractExampleWiseStatistics(labelMatrixPtr->getNumRows(), labelMatrixPtr->getNumCols(),
                                    ruleEvaluationFactoryPtr),
      lossFunctionPtr_(lossFunctionPtr), lapackPtr_(lapackPtr), labelMatrixPtr_(labelMatrixPtr), gradients_(gradients),
      hessians_(hessians), currentScores_(currentScores) {
    // The number of labels
    uint32 numLabels = this->getNumCols();
    // The number of hessians
    uint32 numHessians = linalg::triangularNumber(numLabels);
    // An array that stores the column-wise sums of the matrix of gradients
    totalSumsOfGradients_ = (float64*) malloc(numLabels * sizeof(float64));
    // An array that stores the column-wise sums of the matrix of Hessians
    totalSumsOfHessians_ = (float64*) malloc(numHessians * sizeof(float64));
}

DenseExampleWiseStatisticsImpl::~DenseExampleWiseStatisticsImpl() {
    free(currentScores_);
    free(gradients_);
    free(totalSumsOfGradients_);
    free(hessians_);
    free(totalSumsOfHessians_);
}

void DenseExampleWiseStatisticsImpl::resetCoveredStatistics() {
    uint32 numLabels = this->getNumCols();
    arrays::setToZeros(totalSumsOfGradients_, numLabels);
    uint32 numHessians = linalg::triangularNumber(numLabels);
    arrays::setToZeros(totalSumsOfHessians_, numHessians);
}

void DenseExampleWiseStatisticsImpl::updateCoveredStatistic(uint32 statisticIndex, uint32 weight, bool remove) {
    float64 signedWeight = remove ? -((float64) weight) : weight;
    uint32 numLabels = this->getNumCols();
    uint32 offset = statisticIndex * numLabels;

    // Add the gradients of the example at the given index (weighted by the given weight) to the total sums of
    // gradients...
    for (uint32 c = 0; c < numLabels; c++) {
        totalSumsOfGradients_[c] += (signedWeight * gradients_[offset + c]);
    }

    uint32 numHessians = linalg::triangularNumber(numLabels);
    offset = statisticIndex * numHessians;

    // Add the Hessians of the example at the given index (weighted by the given weight) to the total sums of
    // Hessians...
    for (uint32 c = 0; c < numHessians; c++) {
        totalSumsOfHessians_[c] += (signedWeight * hessians_[offset + c]);
    }
}

std::unique_ptr<IStatisticsSubset> DenseExampleWiseStatisticsImpl::createSubset(
        const FullIndexVector& labelIndices) const {
    std::unique_ptr<IExampleWiseRuleEvaluation> ruleEvaluationPtr = ruleEvaluationFactoryPtr_->create(labelIndices);
    return std::make_unique<StatisticsSubset<FullIndexVector>>(*this, std::move(ruleEvaluationPtr), labelIndices);
}

std::unique_ptr<IStatisticsSubset> DenseExampleWiseStatisticsImpl::createSubset(
        const PartialIndexVector& labelIndices) const {
    std::unique_ptr<IExampleWiseRuleEvaluation> ruleEvaluationPtr = ruleEvaluationFactoryPtr_->create(labelIndices);
    return std::make_unique<StatisticsSubset<PartialIndexVector>>(*this, std::move(ruleEvaluationPtr), labelIndices);
}

void DenseExampleWiseStatisticsImpl::applyPrediction(uint32 statisticIndex, const FullPrediction& prediction) {
    uint32 numLabels = this->getNumCols();
    uint32 numHessians = linalg::triangularNumber(numLabels);
    uint32 offset = statisticIndex * numLabels;
    uint32 numPredictions = prediction.getNumElements();
    FullPrediction::const_iterator valueIterator = prediction.cbegin();

    // Traverse the labels for which the new rule predicts to update the scores that are currently predicted for the
    // example at the given index...
    for (uint32 c = 0; c < numPredictions; c++) {
        currentScores_[offset + c] += valueIterator[c];
    }

    // Update the gradients and Hessians for the example at the given index...
    lossFunctionPtr_->calculateGradientsAndHessians(*labelMatrixPtr_, statisticIndex, &currentScores_[offset],
                                                    &gradients_[offset], &hessians_[statisticIndex * numHessians]);
}

void DenseExampleWiseStatisticsImpl::applyPrediction(uint32 statisticIndex, const PartialPrediction& prediction) {
    uint32 numLabels = this->getNumCols();
    uint32 numHessians = linalg::triangularNumber(numLabels);
    uint32 offset = statisticIndex * numLabels;
    uint32 numPredictions = prediction.getNumElements();
    PartialPrediction::const_iterator valueIterator = prediction.cbegin();
    PartialPrediction::index_const_iterator indexIterator = prediction.indices_cbegin();

    // Traverse the labels for which the new rule predicts to update the scores that are currently predicted for the
    // example at the given index...
    for (uint32 c = 0; c < numPredictions; c++) {
        uint32 l = indexIterator[c];
        currentScores_[offset + l] += valueIterator[c];
    }

    // Update the gradients and Hessians for the example at the given index...
    lossFunctionPtr_->calculateGradientsAndHessians(*labelMatrixPtr_, statisticIndex, &currentScores_[offset],
                                                    &gradients_[offset], &hessians_[statisticIndex * numHessians]);
}

std::unique_ptr<AbstractStatistics::IHistogramBuilder> DenseExampleWiseStatisticsImpl::buildHistogram(
        uint32 numBins) const {
    return std::make_unique<HistogramBuilder>(*this, numBins);
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
    // The number of examples
    uint32 numExamples = labelMatrixPtr_->getNumRows();
    // The number of labels
    uint32 numLabels = labelMatrixPtr_->getNumCols();
    // The number of hessians
    uint32 numHessians = linalg::triangularNumber(numLabels);
    // A matrix that stores the gradients for each example
    float64* gradients = (float64*) malloc(numExamples * numLabels * sizeof(float64));
    // A matrix that stores the Hessians for each example
    float64* hessians = (float64*) malloc(numExamples * numHessians * sizeof(float64));
    // A matrix that stores the currently predicted scores for each example and label
    float64* currentScores = (float64*) malloc(numExamples * numLabels * sizeof(float64));

    for (uint32 r = 0; r < numExamples; r++) {
        uint32 offset = r * numLabels;

        for (uint32 c = 0; c < numLabels; c++) {
            // Store the score that is initially predicted for the current example and label...
            currentScores[offset + c] = 0;
        }

        // Calculate the initial gradients and Hessians for the current example...
        lossFunctionPtr_->calculateGradientsAndHessians(*labelMatrixPtr_, r, &currentScores[offset], &gradients[offset],
                                                        &hessians[r * numHessians]);
    }

    return std::make_unique<DenseExampleWiseStatisticsImpl>(lossFunctionPtr_, ruleEvaluationFactoryPtr_, lapackPtr_,
                                                            labelMatrixPtr_, gradients, hessians, currentScores);
}
