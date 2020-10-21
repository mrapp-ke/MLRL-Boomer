#include "example_wise_statistics.h"
#include "linalg.h"
#include <cstdlib>

using namespace boosting;


AbstractExampleWiseStatistics::AbstractExampleWiseStatistics(
        uint32 numStatistics, uint32 numLabels,
        std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr)
    : AbstractGradientStatistics(numStatistics, numLabels), ruleEvaluationFactoryPtr_(ruleEvaluationFactoryPtr) {

}

void AbstractExampleWiseStatistics::setRuleEvaluationFactory(
        std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr) {
    ruleEvaluationFactoryPtr_ = ruleEvaluationFactoryPtr;
}

DenseExampleWiseStatisticsImpl::StatisticsSubsetImpl::StatisticsSubsetImpl(
        const DenseExampleWiseStatisticsImpl& statistics, std::unique_ptr<IExampleWiseRuleEvaluation> ruleEvaluationPtr,
        uint32 numPredictions, const uint32* labelIndices)
    : statistics_(statistics), ruleEvaluationPtr_(std::move(ruleEvaluationPtr)), numPredictions_(numPredictions),
      labelIndices_(labelIndices) {
    sumsOfGradients_ = (float64*) malloc(numPredictions * sizeof(float64));
    arrays::setToZeros(sumsOfGradients_, numPredictions);
    accumulatedSumsOfGradients_ = nullptr;
    uint32 numHessians = linalg::triangularNumber(numPredictions);
    sumsOfHessians_ = (float64*) malloc(numHessians * sizeof(float64));
    arrays::setToZeros(sumsOfHessians_, numHessians);
    accumulatedSumsOfHessians_ = nullptr;
    prediction_ = nullptr;
    labelWisePrediction_ = nullptr;
    tmpGradients_ = nullptr;
    tmpHessians_ = nullptr;
    dsysvTmpArray1_ = nullptr;
    dsysvTmpArray2_ = nullptr;
    dsysvTmpArray3_ = nullptr;
    dspmvTmpArray_ = nullptr;
}

DenseExampleWiseStatisticsImpl::StatisticsSubsetImpl::~StatisticsSubsetImpl() {
    free(sumsOfGradients_);
    free(accumulatedSumsOfGradients_);
    free(sumsOfHessians_);
    free(accumulatedSumsOfHessians_);
    delete prediction_;
    delete labelWisePrediction_;
    free(tmpGradients_);
    free(tmpHessians_);
    free(dsysvTmpArray1_);
    free(dsysvTmpArray2_);
    free(dsysvTmpArray3_);
    free(dspmvTmpArray_);
}

void DenseExampleWiseStatisticsImpl::StatisticsSubsetImpl::addToSubset(uint32 statisticIndex, uint32 weight) {
    // Add the gradients and Hessians of the example at the given index (weighted by the given weight) to the current
    // sum of gradients and Hessians...
    uint32 numLabels = statistics_.getNumCols();
    uint32 offsetGradients = statisticIndex * numLabels;
    uint32 offsetHessians = statisticIndex * linalg::triangularNumber(numLabels);
    uint32 i = 0;

    for (uint32 c = 0; c < numPredictions_; c++) {
        uint32 l = labelIndices_ != nullptr ? labelIndices_[c] : c;
        sumsOfGradients_[c] += (weight * statistics_.gradients_[offsetGradients + l]);
        uint32 triangularNumber = linalg::triangularNumber(l);

        for (uint32 c2 = 0; c2 < c + 1; c2++) {
            uint32 l2 = triangularNumber + (labelIndices_ != nullptr ? labelIndices_[c2] : c2);
            sumsOfHessians_[i] += (weight * statistics_.hessians_[offsetHessians + l2]);
            i++;
        }
    }
}

void DenseExampleWiseStatisticsImpl::StatisticsSubsetImpl::resetSubset() {
    uint32 numHessians = linalg::triangularNumber(numPredictions_);

    // Allocate arrays for storing the accumulated sums of gradients and Hessians, if necessary...
    if (accumulatedSumsOfGradients_ == nullptr) {
        accumulatedSumsOfGradients_ = (float64*) malloc(numPredictions_ * sizeof(float64));
        arrays::setToZeros(accumulatedSumsOfGradients_, numPredictions_);
        accumulatedSumsOfHessians_ = (float64*) malloc(numHessians * sizeof(float64));
        arrays::setToZeros(accumulatedSumsOfHessians_, numHessians);
    }

    // Reset the sum of gradients and Hessians for each label to zero and add it to the accumulated sums of gradients
    // and Hessians...
    for (uint32 c = 0; c < numPredictions_; c++) {
        accumulatedSumsOfGradients_[c] += sumsOfGradients_[c];
        sumsOfGradients_[c] = 0;
    }

    for (uint32 c = 0; c < numHessians; c++) {
        accumulatedSumsOfHessians_[c] += sumsOfHessians_[c];
        sumsOfHessians_[c] = 0;
    }
}

const LabelWiseEvaluatedPrediction& DenseExampleWiseStatisticsImpl::StatisticsSubsetImpl::calculateLabelWisePrediction(
        bool uncovered, bool accumulated) {
    float64* sumsOfGradients = accumulated ? accumulatedSumsOfGradients_ : sumsOfGradients_;
    float64* sumsOfHessians = accumulated ? accumulatedSumsOfHessians_ : sumsOfHessians_;

    if (labelWisePrediction_ == nullptr) {
        labelWisePrediction_ = new LabelWiseEvaluatedPrediction(numPredictions_);
    }

    ruleEvaluationPtr_->calculateLabelWisePrediction(labelIndices_, statistics_.totalSumsOfGradients_, sumsOfGradients,
                                                     statistics_.totalSumsOfHessians_, sumsOfHessians, uncovered,
                                                     *labelWisePrediction_);
    return *labelWisePrediction_;
}

const EvaluatedPrediction& DenseExampleWiseStatisticsImpl::StatisticsSubsetImpl::calculateExampleWisePrediction(
        bool uncovered, bool accumulated) {
    float64* sumsOfGradients = accumulated ? accumulatedSumsOfGradients_ : sumsOfGradients_;
    float64* sumsOfHessians = accumulated ? accumulatedSumsOfHessians_ : sumsOfHessians_;

    // To avoid array recreation each time this function is called, the temporary arrays are only initialized if they
    // have not been initialized yet
    if (prediction_ == nullptr) {
        prediction_ = new EvaluatedPrediction(numPredictions_);
        tmpGradients_ = (float64*) malloc(numPredictions_ * sizeof(float64));
        uint32 numHessians = linalg::triangularNumber(numPredictions_);
        tmpHessians_ = (float64*) malloc(numHessians * sizeof(float64));
        dsysvTmpArray1_ = (float64*) malloc(numPredictions_ * numPredictions_ * sizeof(float64));
        dsysvTmpArray2_ = (int*) malloc(numPredictions_ * sizeof(int));
        dspmvTmpArray_ = (float64*) malloc(numPredictions_ * sizeof(float64));

        // Query the optimal "lwork" parameter to be used by LAPACK'S DSYSV routine...
        dsysvLwork_ = statistics_.lapackPtr_->queryDsysvLworkParameter(dsysvTmpArray1_, prediction_->begin(),
                                                                       numPredictions_);
        dsysvTmpArray3_ = (double*) malloc(dsysvLwork_ * sizeof(double));
    }

    ruleEvaluationPtr_->calculateExampleWisePrediction(labelIndices_, statistics_.totalSumsOfGradients_,
                                                       sumsOfGradients, statistics_.totalSumsOfHessians_,
                                                       sumsOfHessians, tmpGradients_, tmpHessians_, dsysvLwork_,
                                                       dsysvTmpArray1_, dsysvTmpArray2_, dsysvTmpArray3_,
                                                       dspmvTmpArray_, uncovered, *prediction_);
    return *prediction_;
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

std::unique_ptr<IStatisticsSubset> DenseExampleWiseStatisticsImpl::createSubset(uint32 numLabelIndices,
                                                                                const uint32* labelIndices) const {
    uint32 numLabels = this->getNumCols();
    uint32 numPredictions = labelIndices == nullptr ? numLabels : numLabelIndices;
    std::unique_ptr<IExampleWiseRuleEvaluation> ruleEvaluationPtr = ruleEvaluationFactoryPtr_->create();
    return std::make_unique<DenseExampleWiseStatisticsImpl::StatisticsSubsetImpl>(*this, std::move(ruleEvaluationPtr),
                                                                                  numPredictions, labelIndices);
}

void DenseExampleWiseStatisticsImpl::applyPrediction(uint32 statisticIndex, const Prediction& prediction) {
    uint32 numLabels = this->getNumCols();
    uint32 numPredictions = prediction.numPredictions_;
    const uint32* labelIndices = prediction.labelIndices_;
    const float64* predictedScores = prediction.predictedScores_;
    uint32 offset = statisticIndex * numLabels;
    uint32 numHessians = linalg::triangularNumber(numLabels);

    // Traverse the labels for which the new rule predicts to update the scores that are currently predicted for the
    // example at the given index...
    for (uint32 c = 0; c < numPredictions; c++) {
        uint32 l = labelIndices != nullptr ? labelIndices[c] : c;
        currentScores_[offset + l] += predictedScores[c];
    }

    // Update the gradients and Hessians for the example at the given index...
    lossFunctionPtr_->calculateGradientsAndHessians(*labelMatrixPtr_, statisticIndex, &currentScores_[offset],
                                                    &gradients_[offset], &hessians_[statisticIndex * numHessians]);
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
