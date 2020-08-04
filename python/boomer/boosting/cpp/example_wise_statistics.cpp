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
    hessians_ = hessians;
    totalSumsOfHessians_ = totalSumsOfHessians;
}

ExampleWiseRefinementSearchImpl::~ExampleWiseRefinementSearchImpl() {
    // TODO
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
