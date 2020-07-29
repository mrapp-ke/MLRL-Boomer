#include "label_wise_statistics.h"
#include <stdlib.h>
#include <cstddef>

using namespace statistics;


LabelWiseRefinementSearchImpl::LabelWiseRefinementSearchImpl(
        rule_evaluation::LabelWiseRuleEvaluationImpl* ruleEvaluation, intp numLabels, const intp* labelIndices,
        AbstractLabelMatrix* labelMatrix, const float64* uncoveredLabels, const uint8* minorityLabels,
        const float64* confusionMatricesTotal, const float64* confusionMatricesSubset) {
    ruleEvaluation_ = ruleEvaluation;
    numLabels_ = numLabels;
    labelIndices_ = labelIndices;
    labelMatrix_ = labelMatrix;
    uncoveredLabels_ = uncoveredLabels;
    minorityLabels_ = minorityLabels;
    confusionMatricesTotal_ = confusionMatricesTotal;
    confusionMatricesSubset_ = confusionMatricesSubset;
    float64* confusionMatricesCovered = (float64*) malloc(numLabels * 4 * sizeof(float64));

    for (intp i = 0; i < numLabels * 4; i++) {
        confusionMatricesCovered_[i] = 0;
    }

    confusionMatricesCovered_ = confusionMatricesCovered;
    accumulatedConfusionMatricesCovered_ = NULL;
}

LabelWiseRefinementSearchImpl::~LabelWiseRefinementSearchImpl() {
    free(confusionMatricesCovered_);
    free(accumulatedConfusionMatricesCovered_);
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
