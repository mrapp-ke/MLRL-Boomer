#include "example_wise_statistics.h"
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
    // TODO
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
