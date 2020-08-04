#include "example_wise_statistics.h"
#include "linalg.h"
#include <stdlib.h>
#include <cstddef>

using namespace boosting;


ExampleWiseRefinementSearchImpl::ExampleWiseRefinementSearchImpl(ExampleWiseRuleEvaluationImpl* ruleEvaluation,
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
    intp offset = statisticIndex * numLabels_;
    intp i = 0;

    for (intp c = 0; c < numPredictions_; c++) {
        intp l = labelIndices_ != NULL ? labelIndices_[c] : c;
        intp triangularNumber = linalg::triangularNumber(l)
        sumsOfGradients_[c] += (weight * gradients_[offset + l]);

        for (intp c2 = 0; c2 < c + 1; c2++) {
            intp l2 = triangularNumber + (labelIndices_ != NULL ? labelIndices_[c2] : c2);
            sumsOfHessians_[i] += (weight * hessians_[offset + l2]);
            i++;
        }
    }
}

void ExampleWiseRefinementSearchImpl::resetSearch() {
    // TODO
}

LabelWisePrediction* ExampleWiseRefinementSearchImpl::calculateLabelWisePrediction(bool uncovered, bool accumulated) {
    // TODO
    return NULL;
}

Prediction* ExampleWiseRefinementSearchImpl::calculateExampleWisePrediction(bool uncovered, bool accumulated) {
    // TODO
    return NULL;
}
