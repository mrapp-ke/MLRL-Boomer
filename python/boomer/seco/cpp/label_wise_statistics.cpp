#include "label_wise_statistics.h"
#include "heuristics.h"
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
    confusionMatricesCovered_ = mallocFloat64(numLabels, 4);
    setToZeros(confusionMatricesCovered_, numLabels, 4);
    accumulatedConfusionMatricesCovered_ = NULL;
    float64* predictedScores = mallocFloat64(numLabels);
    float64* qualityScores = mallocFloat64(numLabels);
    prediction_ = new rule_evaluation::LabelWisePrediction(numLabels, predictedScores, qualityScores, 0);
}

LabelWiseRefinementSearchImpl::~LabelWiseRefinementSearchImpl() {
    free(confusionMatricesCovered_);
    free(accumulatedConfusionMatricesCovered_);
    delete prediction_;
}

void LabelWiseRefinementSearchImpl::updateSearch(intp statisticIndex, uint32 weight) {
    intp numTotalLabels = labelMatrix_->numLabels_;
    intp offset = statisticIndex * numTotalLabels;

    for (intp c = 0; c < numLabels_; c++) {
        intp l = labelIndices_ != NULL ? labelIndices_[c] : c;

        // Only uncovered labels must be considered...
        if (uncoveredLabels_[offset + l] > 0) {
            // Add the current example and label to the confusion matrix for the current label...
            uint8 trueLabel = labelMatrix_->getLabel(statisticIndex, l);
            uint8 predictedLabel = minorityLabels_[l];
            intp element = heuristics::getConfusionMatrixElement(trueLabel, predictedLabel);
            confusionMatricesCovered_[c * 4 + element] += weight;
        }
    }
}

void LabelWiseRefinementSearchImpl::resetSearch() {
    if (accumulatedConfusionMatricesCovered_ == NULL) {
        accumulatedConfusionMatricesCovered_ = mallocFloat64(numLabels_, 4);
        setToZeros(accumulatedConfusionMatricesCovered_, numLabels_, 4);
    }

    for (intp c = 0; c < numLabels_; c++) {
        intp offset = c * 4;

        for (intp i = 0; i < 4; i++) {
            intp j = offset + i;
            accumulatedConfusionMatricesCovered_[j] += confusionMatricesCovered_[j];
            confusionMatricesCovered_[j] = 0;
        }
    }
}

rule_evaluation::LabelWisePrediction* LabelWiseRefinementSearchImpl::calculateLabelWisePrediction(bool uncovered,
                                                                                                  bool accumulated) {
    float64* confusionMatricesCovered = accumulated ? accumulatedConfusionMatricesCovered_ : confusionMatricesCovered_;
    ruleEvaluation_->calculateLabelWisePrediction(labelIndices_, minorityLabels_, confusionMatricesTotal_,
                                                  confusionMatricesSubset_, confusionMatricesCovered, uncovered,
                                                  prediction_);
    return prediction_;
}
