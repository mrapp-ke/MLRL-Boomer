#include "label_wise_statistics.h"

using namespace boosting;


LabelWiseRefinementSearchImpl::LabelWiseRefinementSearchImpl(LabelWiseRuleEvaluationImpl* ruleEvaluation,
                                                             intp numPredictions, const intp* labelIndices,
                                                             const float64* gradients,
                                                             const float64* totalSumsOfGradients,
                                                             const float64* hessians,
                                                             const float64* totalSumsOfHessians) {
    ruleEvaluation_ = ruleEvaluation;
    numPredictions_ = numPredictions;
    labelIndices_ = labelIndices;
    gradients_ = gradients;
    totalSumsOfGradients_ = totalSumsOfGradients;
    hessians_ = hessians;
    totalSumsOfHessians_ = totalSumsOfHessians;
    float64* predictedScores = (float64*) malloc(numPredictions * sizeof(float64));
    float64* qualityScores = (float64*) malloc(numPredictions * sizeof(float64));
    prediction_ = new LabelWisePrediction(numPredictions, predictedScores, qualityScores, 0)
}

LabelWiseRefinementSearchImpl::~LabelWiseRefinementSearchImpl() {
    // TODO
    delete prediction_;
}

void LabelWiseRefinementSearchImpl::updateSearch(intp statisticIndex, uint32 weight) {
    // TODO
}

void LabelWiseRefinementSearchImpl::resetSearch() {
    // TODO
}

LabelWisePrediction* LabelWiseRefinementSearchImpl::calculateLabelWisePrediction(bool uncovered, bool accumulated) {
    // TODO
    return NULL;
}
