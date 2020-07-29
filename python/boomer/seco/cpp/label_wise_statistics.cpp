#include "label_wise_statistics.h"
#include <cstddef>

using namespace statistics;


LabelWiseRefinementSearchImpl::LabelWiseRefinementSearchImpl(
        rule_evaluation::LabelWiseRuleEvaluationImpl* ruleEvaluation, intp numLabels, const intp* labelIndices,
        statistics::AbstractLabelMatrix* labelMatrix, const float64* uncoveredLabels, const uint8* minorityLabels,
        const float64* confusionMatricesTotal, const float64* confusionMatricesSubset) {
    ruleEvaluation_ = ruleEvaluation;
    numLabels_ = numLabels;
    labelIndices_ = labelIndices;
    labelMatrix_ = labelMatrix;
    uncoveredLabels_ = uncoveredLabels;
    minorityLabels_ = minorityLabels;
    confusionMatricesTotal_ = confusionMatricesTotal;
    confusionMatricesSubset_ = confusionMatricesSubset;
}

LabelWiseRefinementSearchImpl::~LabelWiseRefinementSearchImpl() {
    // TODO
}

void LabelWiseRefinementSearchImpl::updateSearch(intp statisticIndex, uint32 weight) {
    // TODO
}

void LabelWiseRefinementSearchImpl::resetSearch() {
    // TODO
}

rule_evaluation::LabelWisePrediction* LabelWiseRefinementSearchImpl::calculateLabelWisePrediction(bool uncovered,
                                                                                                  bool accumulated) {
    // TODO
    return NULL;
}
