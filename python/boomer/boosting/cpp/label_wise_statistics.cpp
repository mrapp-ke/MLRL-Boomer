#include "label_wise_statistics.h"
#include <stdlib.h>
#include <cstddef>

using namespace boosting;


DenseLabelWiseRefinementSearchImpl::DenseLabelWiseRefinementSearchImpl(
        std::shared_ptr<AbstractLabelWiseRuleEvaluation> ruleEvaluationPtr, intp numPredictions,
        const intp* labelIndices, intp numLabels, const float64* gradients, const float64* totalSumsOfGradients,
        const float64* hessians, const float64* totalSumsOfHessians) {
    ruleEvaluationPtr_ = ruleEvaluationPtr;
    numPredictions_ = numPredictions;
    labelIndices_ = labelIndices;
    numLabels_ = numLabels;
    gradients_ = gradients;
    totalSumsOfGradients_ = totalSumsOfGradients;
    float64* sumsOfGradients = (float64*) malloc(numPredictions * sizeof(float64));
    arrays::setToZeros(sumsOfGradients, numPredictions);
    sumsOfGradients_ = sumsOfGradients;
    accumulatedSumsOfGradients_ = NULL;
    hessians_ = hessians;
    totalSumsOfHessians_ = totalSumsOfHessians;
    float64* sumsOfHessians = (float64*) malloc(numPredictions * sizeof(float64));
    arrays::setToZeros(sumsOfHessians, numPredictions);
    sumsOfHessians_ = sumsOfHessians;
    accumulatedSumsOfHessians_ = NULL;
    float64* predictedScores = (float64*) malloc(numPredictions * sizeof(float64));
    float64* qualityScores = (float64*) malloc(numPredictions * sizeof(float64));
    prediction_ = new LabelWisePredictionCandidate(numPredictions, NULL, predictedScores, qualityScores, 0);
}

DenseLabelWiseRefinementSearchImpl::~DenseLabelWiseRefinementSearchImpl() {
    free(sumsOfGradients_);
    free(accumulatedSumsOfGradients_);
    free(sumsOfHessians_);
    free(accumulatedSumsOfHessians_);
    delete prediction_;
}

void DenseLabelWiseRefinementSearchImpl::updateSearch(intp statisticIndex, uint32 weight) {
    // For each label, add the gradient and Hessian of the example at the given index (weighted by the given weight) to
    // the current sum of gradients and Hessians...
    intp offset = statisticIndex * numLabels_;

    for (intp c = 0; c < numPredictions_; c++) {
        intp l = labelIndices_ != NULL ? labelIndices_[c] : c;
        intp i = offset + l;
        sumsOfGradients_[c] += (weight * gradients_[i]);
        sumsOfHessians_[c] += (weight * hessians_[i]) ;
    }
}

void DenseLabelWiseRefinementSearchImpl::resetSearch() {
    // Allocate arrays for storing the accumulated sums of gradients and Hessians, if necessary...
    if (accumulatedSumsOfGradients_ == NULL) {
        accumulatedSumsOfGradients_ = (float64*) malloc(numPredictions_ * sizeof(float64));
        arrays::setToZeros(accumulatedSumsOfGradients_, numPredictions_);
        accumulatedSumsOfHessians_ = (float64*) malloc(numPredictions_ * sizeof(float64));
        arrays::setToZeros(accumulatedSumsOfHessians_, numPredictions_);
    }

    // Reset the sum of gradients and Hessians for each label to zero and add it to the accumulated sums of gradients
    // and hessians...
    for (intp c = 0; c < numPredictions_; c++) {
        accumulatedSumsOfGradients_[c] += sumsOfGradients_[c];
        sumsOfGradients_[c] = 0;
        accumulatedSumsOfHessians_[c] += sumsOfHessians_[c];
        sumsOfHessians_[c] = 0;
    }
}

LabelWisePredictionCandidate* DenseLabelWiseRefinementSearchImpl::calculateLabelWisePrediction(bool uncovered,
                                                                                               bool accumulated) {
    float64* sumsOfGradients = accumulated ? accumulatedSumsOfGradients_ : sumsOfGradients_;
    float64* sumsOfHessians = accumulated ? accumulatedSumsOfHessians_ : sumsOfHessians_;
    ruleEvaluationPtr_.get()->calculateLabelWisePrediction(labelIndices_, totalSumsOfGradients_, sumsOfGradients,
                                                           totalSumsOfHessians_, sumsOfHessians, uncovered,
                                                           prediction_);
    return prediction_;
}

AbstractLabelWiseStatistics::AbstractLabelWiseStatistics(intp numStatistics)
    : AbstractGradientStatistics(numStatistics) {

}

DenseLabelWiseStatisticsImpl::DenseLabelWiseStatisticsImpl(
        std::shared_ptr<AbstractLabelWiseLoss> lossFunctionPtr,
        std::shared_ptr<AbstractLabelWiseRuleEvaluation> ruleEvaluationPtr,
        std::shared_ptr<AbstractRandomAccessLabelMatrix> labelMatrixPtr, float64* gradients, float64* hessians,
        float64* currentScores)
    : AbstractLabelWiseStatistics(labelMatrixPtr.get()->numExamples_) {
    lossFunctionPtr_ = lossFunctionPtr;
    ruleEvaluationPtr_ = ruleEvaluationPtr;
    labelMatrixPtr_ = labelMatrixPtr;
    gradients_ = gradients;
    hessians_ = hessians;
    currentScores_ = currentScores;
    // The number of labels
    intp numLabels = labelMatrixPtr.get()->numLabels_;
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
    intp numLabels = labelMatrixPtr_.get()->numLabels_;
    arrays::setToZeros(totalSumsOfGradients_, numLabels);
    arrays::setToZeros(totalSumsOfHessians_, numLabels);
}

void DenseLabelWiseStatisticsImpl::updateCoveredStatistic(intp statisticIndex, uint32 weight, bool remove) {
    intp numLabels = labelMatrixPtr_.get()->numLabels_;
    intp offset = statisticIndex * numLabels;
    float64 signedWeight = remove ? -((float64) weight) : weight;

    // For each label, add the gradient and Hessian of the example at the given index (weighted by the given weight) to
    // the total sums of gradients and Hessians...
    for (intp c = 0; c < numLabels; c++) {
        intp i = offset + c;
        totalSumsOfGradients_[c] += (signedWeight * gradients_[i]);
        totalSumsOfHessians_[c] += (signedWeight * hessians_[i]);
    }
}

AbstractRefinementSearch* DenseLabelWiseStatisticsImpl::beginSearch(intp numLabelIndices, const intp* labelIndices) {
    intp numLabels = labelMatrixPtr_.get()->numLabels_;
    intp numPredictions = labelIndices == NULL ? numLabels : numLabelIndices;
    return new DenseLabelWiseRefinementSearchImpl(ruleEvaluationPtr_, numPredictions, labelIndices, numLabels,
                                                  gradients_, totalSumsOfGradients_, hessians_, totalSumsOfHessians_);
}

void DenseLabelWiseStatisticsImpl::applyPrediction(intp statisticIndex, Prediction* prediction) {
    AbstractLabelWiseLoss* lossFunction = lossFunctionPtr_.get();
    intp numPredictions = prediction->numPredictions_;
    const intp* labelIndices = prediction->labelIndices_;
    const float64* predictedScores = prediction->predictedScores_;
    intp numLabels = labelMatrixPtr_.get()->numLabels_;
    intp offset = statisticIndex * numLabels;

    // Only the labels that are predicted by the new rule must be considered...
    for (intp c = 0; c < numPredictions; c++) {
        intp l = labelIndices != NULL ? labelIndices[c] : c;
        float64 predictedScore = predictedScores[c];
        intp i = offset + l;

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
    intp numExamples = labelMatrix->numExamples_;
    // The number of labels
    intp numLabels = labelMatrix->numLabels_;
    // A matrix that stores the gradients for each example and label
    float64* gradients = (float64*) malloc(numExamples * numLabels * sizeof(float64));
    // A matrix that stores the Hessians for each example and label
    float64* hessians = (float64*) malloc(numExamples * numLabels * sizeof(float64));
    // A matrix that stores the currently predicted scores for each example and label
    float64* currentScores = (float64*) malloc(numExamples * numLabels * sizeof(float64));

    for (intp r = 0; r < numExamples; r++) {
        intp offset = r * numLabels;

        for (intp c = 0; c < numLabels; c++) {
            intp i = offset + c;

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
