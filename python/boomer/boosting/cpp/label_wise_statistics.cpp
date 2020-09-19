#include "label_wise_statistics.h"
#include <stdlib.h>
#include <cstddef>

using namespace boosting;


DenseLabelWiseStatisticsImpl::StatisticsSubsetImpl::StatisticsSubsetImpl(DenseLabelWiseStatisticsImpl* statistics,
                                                                         uint32 numPredictions,
                                                                         const uint32* labelIndices) {
    statistics_ = statistics;
    numPredictions_ = numPredictions;
    labelIndices_ = labelIndices;
    sumsOfGradients_ = (float64*) malloc(numPredictions * sizeof(float64));
    arrays::setToZeros(sumsOfGradients_, numPredictions);
    accumulatedSumsOfGradients_ = NULL;
    sumsOfHessians_ = (float64*) malloc(numPredictions * sizeof(float64));
    arrays::setToZeros(sumsOfHessians_, numPredictions);
    accumulatedSumsOfHessians_ = NULL;
    float64* predictedScores = (float64*) malloc(numPredictions * sizeof(float64));
    float64* qualityScores = (float64*) malloc(numPredictions * sizeof(float64));
    prediction_ = new LabelWisePredictionCandidate(numPredictions, NULL, predictedScores, qualityScores, 0);
}

DenseLabelWiseStatisticsImpl::StatisticsSubsetImpl::~StatisticsSubsetImpl() {
    free(sumsOfGradients_);
    free(accumulatedSumsOfGradients_);
    free(sumsOfHessians_);
    free(accumulatedSumsOfHessians_);
    delete prediction_;
}

void DenseLabelWiseStatisticsImpl::StatisticsSubsetImpl::addToSubset(uint32 statisticIndex, uint32 weight) {
    // For each label, add the gradient and Hessian of the example at the given index (weighted by the given weight) to
    // the current sum of gradients and Hessians...
    uint32 offset = statisticIndex * statistics_->getNumCols();

    for (uint32 c = 0; c < numPredictions_; c++) {
        uint32 l = labelIndices_ != NULL ? labelIndices_[c] : c;
        uint32 i = offset + l;
        sumsOfGradients_[c] += (weight * statistics_->gradients_[i]);
        sumsOfHessians_[c] += (weight * statistics_->hessians_[i]) ;
    }
}

void DenseLabelWiseStatisticsImpl::StatisticsSubsetImpl::resetSearch() {
    // Allocate arrays for storing the accumulated sums of gradients and Hessians, if necessary...
    if (accumulatedSumsOfGradients_ == NULL) {
        accumulatedSumsOfGradients_ = (float64*) malloc(numPredictions_ * sizeof(float64));
        arrays::setToZeros(accumulatedSumsOfGradients_, numPredictions_);
        accumulatedSumsOfHessians_ = (float64*) malloc(numPredictions_ * sizeof(float64));
        arrays::setToZeros(accumulatedSumsOfHessians_, numPredictions_);
    }

    // Reset the sum of gradients and Hessians for each label to zero and add it to the accumulated sums of gradients
    // and hessians...
    for (uint32 c = 0; c < numPredictions_; c++) {
        accumulatedSumsOfGradients_[c] += sumsOfGradients_[c];
        sumsOfGradients_[c] = 0;
        accumulatedSumsOfHessians_[c] += sumsOfHessians_[c];
        sumsOfHessians_[c] = 0;
    }
}

LabelWisePredictionCandidate* DenseLabelWiseStatisticsImpl::StatisticsSubsetImpl::calculateLabelWisePrediction(
        bool uncovered, bool accumulated) {
    float64* sumsOfGradients = accumulated ? accumulatedSumsOfGradients_ : sumsOfGradients_;
    float64* sumsOfHessians = accumulated ? accumulatedSumsOfHessians_ : sumsOfHessians_;
    statistics_->ruleEvaluationPtr_.get()->calculateLabelWisePrediction(labelIndices_,
                                                                        statistics_->totalSumsOfGradients_,
                                                                        sumsOfGradients,
                                                                        statistics_->totalSumsOfHessians_,
                                                                        sumsOfHessians, uncovered, prediction_);
    return prediction_;
}

AbstractLabelWiseStatistics::AbstractLabelWiseStatistics(
        uint32 numStatistics, uint32 numLabels, std::shared_ptr<AbstractLabelWiseRuleEvaluation> ruleEvaluationPtr)
    : AbstractGradientStatistics(numStatistics, numLabels) {
    this->setRuleEvaluation(ruleEvaluationPtr);
}

void AbstractLabelWiseStatistics::setRuleEvaluation(
        std::shared_ptr<AbstractLabelWiseRuleEvaluation> ruleEvaluationPtr) {
    ruleEvaluationPtr_ = ruleEvaluationPtr;
}

DenseLabelWiseStatisticsImpl::DenseLabelWiseStatisticsImpl(
        std::shared_ptr<AbstractLabelWiseLoss> lossFunctionPtr,
        std::shared_ptr<AbstractLabelWiseRuleEvaluation> ruleEvaluationPtr,
        std::shared_ptr<AbstractRandomAccessLabelMatrix> labelMatrixPtr, float64* gradients, float64* hessians,
        float64* currentScores)
    : AbstractLabelWiseStatistics(labelMatrixPtr.get()->getNumRows(), labelMatrixPtr.get()->getNumCols(),
                                  ruleEvaluationPtr) {
    lossFunctionPtr_ = lossFunctionPtr;
    labelMatrixPtr_ = labelMatrixPtr;
    gradients_ = gradients;
    hessians_ = hessians;
    currentScores_ = currentScores;
    // The number of labels
    uint32 numLabels = this->getNumCols();
    // An array that stores the column-wise sums of the matrix of gradients
    totalSumsOfGradients_ = (float64*) malloc(numLabels * sizeof(float64));
    // An array that stores the column-wise sums of the matrix of hessians
    totalSumsOfHessians_ = (float64*) malloc(numLabels * sizeof(float64));
}

DenseLabelWiseStatisticsImpl::~DenseLabelWiseStatisticsImpl() {
    free(currentScores_);
    free(gradients_);
    free(totalSumsOfGradients_);
    free(hessians_);
    free(totalSumsOfHessians_);
}

void DenseLabelWiseStatisticsImpl::resetCoveredStatistics() {
    uint32 numLabels = this->getNumCols();
    arrays::setToZeros(totalSumsOfGradients_, numLabels);
    arrays::setToZeros(totalSumsOfHessians_, numLabels);
}

void DenseLabelWiseStatisticsImpl::updateCoveredStatistic(uint32 statisticIndex, uint32 weight, bool remove) {
    uint32 numLabels = this->getNumCols();
    uint32 offset = statisticIndex * numLabels;
    float64 signedWeight = remove ? -((float64) weight) : weight;

    // For each label, add the gradient and Hessian of the example at the given index (weighted by the given weight) to
    // the total sums of gradients and Hessians...
    for (uint32 c = 0; c < numLabels; c++) {
        uint32 i = offset + c;
        totalSumsOfGradients_[c] += (signedWeight * gradients_[i]);
        totalSumsOfHessians_[c] += (signedWeight * hessians_[i]);
    }
}

AbstractStatisticsSubset* DenseLabelWiseStatisticsImpl::createSubset(uint32 numLabelIndices,
                                                                     const uint32* labelIndices) {
    uint32 numLabels = this->getNumCols();
    uint32 numPredictions = labelIndices == NULL ? numLabels : numLabelIndices;
    return new DenseLabelWiseStatisticsImpl::StatisticsSubsetImpl(this, numPredictions, labelIndices);
}

void DenseLabelWiseStatisticsImpl::applyPrediction(uint32 statisticIndex, Prediction* prediction) {
    AbstractLabelWiseLoss* lossFunction = lossFunctionPtr_.get();
    uint32 numLabels = this->getNumCols();
    uint32 numPredictions = prediction->numPredictions_;
    const uint32* labelIndices = prediction->labelIndices_;
    const float64* predictedScores = prediction->predictedScores_;
    uint32 offset = statisticIndex * numLabels;

    // Only the labels that are predicted by the new rule must be considered...
    for (uint32 c = 0; c < numPredictions; c++) {
        uint32 l = labelIndices != NULL ? labelIndices[c] : c;
        float64 predictedScore = predictedScores[c];
        uint32 i = offset + l;

        // Update the score that is currently predicted for the current example and label...
        float64 updatedScore = currentScores_[i] + predictedScore;
        currentScores_[i] = updatedScore;

        // Update the gradient and Hessian for the current example and label...
        std::pair<float64, float64> pair = lossFunction->calculateGradientAndHessian(labelMatrixPtr_.get(),
                                                                                     statisticIndex, l, updatedScore);
        gradients_[i] = pair.first;
        hessians_[i] = pair.second;
    }
}

AbstractLabelWiseStatisticsFactory::~AbstractLabelWiseStatisticsFactory() {

}

AbstractLabelWiseStatistics* AbstractLabelWiseStatisticsFactory::create() {
    return NULL;
}

DenseLabelWiseStatisticsFactoryImpl::DenseLabelWiseStatisticsFactoryImpl(
        std::shared_ptr<AbstractLabelWiseLoss> lossFunctionPtr,
        std::shared_ptr<AbstractLabelWiseRuleEvaluation> ruleEvaluationPtr,
        std::shared_ptr<AbstractRandomAccessLabelMatrix> labelMatrixPtr) {
    lossFunctionPtr_ = lossFunctionPtr;
    ruleEvaluationPtr_ = ruleEvaluationPtr;
    labelMatrixPtr_ = labelMatrixPtr;
}

DenseLabelWiseStatisticsFactoryImpl::~DenseLabelWiseStatisticsFactoryImpl() {

}

AbstractLabelWiseStatistics* DenseLabelWiseStatisticsFactoryImpl::create() {
    // Class members
    AbstractLabelWiseLoss* lossFunction = lossFunctionPtr_.get();
    AbstractRandomAccessLabelMatrix* labelMatrix = labelMatrixPtr_.get();
    // The number of examples
    uint32 numExamples = labelMatrix->getNumRows();
    // The number of labels
    uint32 numLabels = labelMatrix->getNumCols();
    // A matrix that stores the gradients for each example and label
    float64* gradients = (float64*) malloc(numExamples * numLabels * sizeof(float64));
    // A matrix that stores the Hessians for each example and label
    float64* hessians = (float64*) malloc(numExamples * numLabels * sizeof(float64));
    // A matrix that stores the currently predicted scores for each example and label
    float64* currentScores = (float64*) malloc(numExamples * numLabels * sizeof(float64));

    for (uint32 r = 0; r < numExamples; r++) {
        uint32 offset = r * numLabels;

        for (uint32 c = 0; c < numLabels; c++) {
            uint32 i = offset + c;

            // Calculate the initial gradient and Hessian for the current example and label...
            std::pair<float64, float64> pair = lossFunction->calculateGradientAndHessian(labelMatrix, r, c, 0);
            gradients[i] = pair.first;
            hessians[i] = pair.second;

            // Store the score that is initially predicted for the current example and label...
            currentScores[i] = 0;
        }
    }

    return new DenseLabelWiseStatisticsImpl(lossFunctionPtr_, ruleEvaluationPtr_, labelMatrixPtr_, gradients, hessians,
                                            currentScores);
}
