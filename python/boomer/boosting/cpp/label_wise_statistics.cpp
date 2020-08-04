#include "label_wise_statistics.h"
#include <stdlib.h>
#include <cstddef>

using namespace boosting;


LabelWiseRefinementSearchImpl::LabelWiseRefinementSearchImpl(LabelWiseRuleEvaluationImpl* ruleEvaluation,
                                                             intp numPredictions, const intp* labelIndices,
                                                             intp numLabels, const float64* gradients,
                                                             const float64* totalSumsOfGradients,
                                                             const float64* hessians,
                                                             const float64* totalSumsOfHessians) {
    ruleEvaluation_ = ruleEvaluation;
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
    prediction_ = new LabelWisePrediction(numPredictions, predictedScores, qualityScores, 0)
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
    intp offset = statistic_index * numLabels_;

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
    ruleEvaluation_->calculateLabelWisePrediction(labelIndices_, totalSumsOfGradients_, sumsOfGradients,
                                                  totalSumsOfHessians_, sumsOfHessians, uncovered, prediction_);
    return prediction_;
}
