#include "example_wise_statistics.h"
#include "linalg.h"
#include <stdlib.h>
#include <cstddef>

using namespace boosting;


ExampleWiseRefinementSearchImpl::ExampleWiseRefinementSearchImpl(
        std::shared_ptr<ExampleWiseRuleEvaluationImpl> ruleEvaluationPtr, intp numPredictions, const intp* labelIndices,
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
    intp numHessians = linalg::triangularNumber(numPredictions);
    float64* sumsOfHessians = (float64*) malloc(numHessians * sizeof(float64));
    arrays::setToZeros(sumsOfHessians, numHessians);
    sumsOfHessians_ = sumsOfHessians;
    accumulatedSumsOfHessians_ = NULL;
    float64* predictedScores = (float64*) malloc(numPredictions * sizeof(float64));
    prediction_ = new LabelWisePrediction(numPredictions, predictedScores, NULL, 0);
}

ExampleWiseRefinementSearchImpl::~ExampleWiseRefinementSearchImpl() {
    free(sumsOfGradients_);
    free(accumulatedSumsOfGradients_);
    free(sumsOfHessians_);
    free(accumulatedSumsOfHessians_);
    delete prediction_;
}

void ExampleWiseRefinementSearchImpl::updateSearch(intp statisticIndex, uint32 weight) {
    // Add the gradients and Hessians of the example at the given index (weighted by the given weight) to the current
    // sum of gradients and Hessians...
    intp offsetGradients = statisticIndex * numLabels_;
    intp offsetHessians = statisticIndex * linalg::triangularNumber(numLabels_);
    intp i = 0;

    for (intp c = 0; c < numPredictions_; c++) {
        intp l = labelIndices_ != NULL ? labelIndices_[c] : c;
        sumsOfGradients_[c] += (weight * gradients_[offsetGradients + l]);
        intp triangularNumber = linalg::triangularNumber(l);

        for (intp c2 = 0; c2 < c + 1; c2++) {
            intp l2 = triangularNumber + (labelIndices_ != NULL ? labelIndices_[c2] : c2);
            sumsOfHessians_[i] += (weight * hessians_[offsetHessians + l2]);
            i++;
        }
    }
}

void ExampleWiseRefinementSearchImpl::resetSearch() {
    intp numHessians = linalg::triangularNumber(numPredictions_);

    // Allocate arrays for storing the accumulated sums of gradients and Hessians, if necessary...
    if (accumulatedSumsOfGradients_ == NULL) {
        accumulatedSumsOfGradients_ = (float64*) malloc(numPredictions_ * sizeof(float64));
        arrays::setToZeros(accumulatedSumsOfGradients_, numPredictions_);
        accumulatedSumsOfHessians_ = (float64*) malloc(numHessians * sizeof(float64));
        arrays::setToZeros(accumulatedSumsOfHessians_, numHessians);
    }

    // Reset the sum of gradients and Hessians for each label to zero and add it to the accumulated sums of gradients
    // and Hessians...
    for (intp c = 0; c < numPredictions_; c++) {
        accumulatedSumsOfGradients_[c] += sumsOfGradients_[c];
        sumsOfGradients_[c] = 0;
    }

    for (intp c = 0; c < numHessians; c++) {
        accumulatedSumsOfHessians_[c] += sumsOfHessians_[c];
        sumsOfHessians_[c] = 0;
    }
}

LabelWisePrediction* ExampleWiseRefinementSearchImpl::calculateLabelWisePrediction(bool uncovered, bool accumulated) {
    float64* sumsOfGradients = accumulated ? accumulatedSumsOfGradients_ : sumsOfGradients_;
    float64* sumsOfHessians = accumulated ? accumulatedSumsOfHessians_ : sumsOfHessians_;
    ruleEvaluationPtr_.get()->calculateLabelWisePrediction(labelIndices_, totalSumsOfGradients_, sumsOfGradients,
                                                           totalSumsOfHessians_, sumsOfHessians, uncovered,
                                                           prediction_);
    return prediction_;
}

Prediction* ExampleWiseRefinementSearchImpl::calculateExampleWisePrediction(bool uncovered, bool accumulated) {
    float64* sumsOfGradients = accumulated ? accumulatedSumsOfGradients_ : sumsOfGradients_;
    float64* sumsOfHessians = accumulated ? accumulatedSumsOfHessians_ : sumsOfHessians_;
    ruleEvaluationPtr_.get()->calculateExampleWisePrediction(labelIndices_, totalSumsOfGradients_, sumsOfGradients,
                                                             totalSumsOfHessians_, sumsOfHessians, uncovered,
                                                             prediction_);
    return prediction_;
}

ExampleWiseStatisticsImpl::ExampleWiseStatisticsImpl(std::shared_ptr<AbstractExampleWiseLoss> lossFunctionPtr,
                                                     std::shared_ptr<ExampleWiseRuleEvaluationImpl> ruleEvaluationPtr) {
    lossFunctionPtr_ = lossFunctionPtr;
    ruleEvaluationPtr_ = ruleEvaluationPtr;
    currentScores_ = NULL;
    gradients_ = NULL;
    totalSumsOfGradients_ = NULL;
    hessians_ = NULL;
    totalSumsOfHessians_ = NULL;
}

ExampleWiseStatisticsImpl::~ExampleWiseStatisticsImpl() {
    free(currentScores_);
    free(gradients_);
    free(totalSumsOfGradients_);
    free(hessians_);
    free(totalSumsOfHessians_);
}

void ExampleWiseStatisticsImpl::applyDefaultPrediction(AbstractLabelMatrix* labelMatrix,
                                                       DefaultPrediction* defaultPrediction) {
    // TODO
}

void ExampleWiseStatisticsImpl::resetCoveredStatistics() {
    // TODO
}

void ExampleWiseStatisticsImpl::updateCoveredStatistic(intp statisticIndex, uint32 weight, bool remove) {
    // TODO
}

AbstractRefinementSearch* ExampleWiseStatisticsImpl::beginSearch(intp numLabelIndices, const intp* labelIndices) {
    // TODO
    return NULL;
}

void ExampleWiseStatisticsImpl::applyPrediction(intp statisticIndex, const intp* labelIndices, HeadCandidate* head) {
    // TODO
}
