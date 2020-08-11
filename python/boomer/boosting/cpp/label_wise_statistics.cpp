#include "label_wise_statistics.h"
#include <stdlib.h>
#include <cstddef>

using namespace boosting;


LabelWiseRefinementSearchImpl::LabelWiseRefinementSearchImpl(
        std::shared_ptr<LabelWiseRuleEvaluationImpl> ruleEvaluationPtr, intp numPredictions, const intp* labelIndices,
        intp numLabels, const float64* gradients, const float64* totalSumsOfGradients, const float64* hessians,
        const float64* totalSumsOfHessians) {
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
    prediction_ = new LabelWisePrediction(numPredictions, predictedScores, qualityScores, 0);
}

LabelWiseRefinementSearchImpl::~LabelWiseRefinementSearchImpl() {
    free(sumsOfGradients_);
    free(accumulatedSumsOfGradients_);
    free(sumsOfHessians_);
    free(accumulatedSumsOfHessians_);
    delete prediction_;
}

void LabelWiseRefinementSearchImpl::updateSearch(intp statisticIndex, uint32 weight) {
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

void LabelWiseRefinementSearchImpl::resetSearch() {
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

LabelWisePrediction* LabelWiseRefinementSearchImpl::calculateLabelWisePrediction(bool uncovered, bool accumulated) {
    float64* sumsOfGradients = accumulated ? accumulatedSumsOfGradients_ : sumsOfGradients_;
    float64* sumsOfHessians = accumulated ? accumulatedSumsOfHessians_ : sumsOfHessians_;
    ruleEvaluationPtr_.get()->calculateLabelWisePrediction(labelIndices_, totalSumsOfGradients_, sumsOfGradients,
                                                           totalSumsOfHessians_, sumsOfHessians, uncovered,
                                                           prediction_);
    return prediction_;
}

LabelWiseStatisticsImpl::LabelWiseStatisticsImpl(std::shared_ptr<AbstractLabelWiseLoss> lossFunctionPtr,
                                                 std::shared_ptr<LabelWiseRuleEvaluationImpl> ruleEvaluationPtr) {
    lossFunctionPtr_ = lossFunctionPtr;
    ruleEvaluationPtr_ = ruleEvaluationPtr;
    currentScores_ = NULL;
    gradients_ = NULL;
    totalSumsOfGradients_ = NULL;
    hessians_ = NULL;
    totalSumsOfHessians_ = NULL;
}

LabelWiseStatisticsImpl::~LabelWiseStatisticsImpl() {
    free(currentScores_);
    free(gradients_);
    free(totalSumsOfGradients_);
    free(hessians_);
    free(totalSumsOfHessians_);
}

void LabelWiseStatisticsImpl::applyDefaultPrediction(std::shared_ptr<AbstractLabelMatrix> labelMatrixPtr,
                                                     DefaultPrediction* defaultPrediction) {
    // Class members
    AbstractLabelWiseLoss* lossFunction = lossFunctionPtr_.get();
    // The number of examples
    intp numExamples = labelMatrixPtr.get()->numExamples_;
    // The number of labels
    intp numLabels = labelMatrixPtr.get()->numLabels_;
    // A matrix that stores the currently predicted scores for each example and label
    float64* currentScores = (float64*) malloc(numExamples * numLabels * sizeof(float64));
    // A matrix that stores the gradients for each example and label
    float64* gradients = (float64*) malloc(numExamples * numLabels * sizeof(float64));
    // An array that stores the column-wise sums of the matrix of gradients
    float64* totalSumsOfGradients = (float64*) malloc(numLabels * sizeof(float64));
    // A matrix that stores the Hessians for each example and label
    float64* hessians = (float64*) malloc(numExamples * numLabels * sizeof(float64));
    // An array that stores the column-wise sums of the matrix of hessians
    float64* totalSumsOfHessians = (float64*) malloc(numLabels * sizeof(float64));
    // An array that stores the predictions of the default rule or NULL, if no default rule is used
    float64* predictedScores = defaultPrediction != NULL ? defaultPrediction->predictedScores_ : NULL;

    for (intp c = 0; c < numLabels; c++) {
        float64 predictedScore = predictedScores != NULL ? predictedScores[c] : 0;

        for (intp r = 0; r < numExamples; r++) {
            intp i = r * numLabels + c;

            // Calculate the gradient and Hessian for the current example and label...
            std::pair<float64, float64> pair = lossFunction->calculateGradientAndHessian(labelMatrixPtr.get(), r, c,
                                                                                         predictedScore);
            gradients[i] = pair.first;
            hessians[i] = pair.second;

            // Store the score that is predicted by the default rule for the current example and label...
            currentScores[i] = predictedScore;
        }
    }

    // Store class members...
    labelMatrixPtr_ = labelMatrixPtr;
    currentScores_ = currentScores;
    gradients_ = gradients;
    totalSumsOfGradients_ = totalSumsOfGradients;
    hessians_ = hessians;
    totalSumsOfHessians_ = totalSumsOfHessians;
}

void LabelWiseStatisticsImpl::resetCoveredStatistics() {
    intp numLabels = labelMatrixPtr_.get()->numLabels_;
    arrays::setToZeros(totalSumsOfGradients_, numLabels);
    arrays::setToZeros(totalSumsOfHessians_, numLabels);
}

void LabelWiseStatisticsImpl::updateCoveredStatistic(intp statisticIndex, uint32 weight, bool remove) {
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

AbstractRefinementSearch* LabelWiseStatisticsImpl::beginSearch(intp numLabelIndices, const intp* labelIndices) {
    intp numLabels = labelMatrixPtr_.get()->numLabels_;
    intp numPredictions = labelIndices == NULL ? numLabels : numLabelIndices;
    return new LabelWiseRefinementSearchImpl(ruleEvaluationPtr_, numPredictions, labelIndices, numLabels, gradients_,
                                             totalSumsOfGradients_, hessians_, totalSumsOfHessians_);
}

void LabelWiseStatisticsImpl::applyPrediction(intp statisticIndex, const intp* labelIndices, HeadCandidate* head) {
    AbstractLabelWiseLoss* lossFunction = lossFunctionPtr_.get();
    intp numPredictions = head->numPredictions_;
    float64* predictedScores = head->predictedScores_;
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
