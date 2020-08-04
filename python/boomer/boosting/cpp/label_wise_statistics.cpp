#include "label_wise_statistics.h"

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
    // TODO
}

LabelWisePrediction* LabelWiseRefinementSearchImpl::calculateLabelWisePrediction(bool uncovered, bool accumulated) {
    // TODO
    return NULL;
}
