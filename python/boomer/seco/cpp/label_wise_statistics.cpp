#include "label_wise_statistics.h"
#include "heuristics.h"
#include <stdlib.h>
#include <cstddef>

using namespace seco;


LabelWiseRefinementSearchImpl::LabelWiseRefinementSearchImpl(LabelWiseRuleEvaluationImpl* ruleEvaluation,
                                                             intp numPredictions, const intp* labelIndices,
                                                             AbstractLabelMatrix* labelMatrix,
                                                             const float64* uncoveredLabels,
                                                             const uint8* minorityLabels,
                                                             const float64* confusionMatricesTotal,
                                                             const float64* confusionMatricesSubset) {
    ruleEvaluation_ = ruleEvaluation;
    numPredictions_ = numPredictions;
    labelIndices_ = labelIndices;
    labelMatrix_ = labelMatrix;
    uncoveredLabels_ = uncoveredLabels;
    minorityLabels_ = minorityLabels;
    confusionMatricesTotal_ = confusionMatricesTotal;
    confusionMatricesSubset_ = confusionMatricesSubset;
    confusionMatricesCovered_ = (float64*) malloc(numPredictions * 4 * sizeof(float64));
    arrays::setToZeros(confusionMatricesCovered_, numPredictions, 4);
    accumulatedConfusionMatricesCovered_ = NULL;
    float64* predictedScores = (float64*) malloc(numPredictions * sizeof(float64));
    float64* qualityScores = (float64*) malloc(numPredictions * sizeof(float64));
    prediction_ = new LabelWisePrediction(numPredictions, predictedScores, qualityScores, 0);
}

LabelWiseRefinementSearchImpl::~LabelWiseRefinementSearchImpl() {
    free(confusionMatricesCovered_);
    free(accumulatedConfusionMatricesCovered_);
    delete prediction_;
}

void LabelWiseRefinementSearchImpl::updateSearch(intp statisticIndex, uint32 weight) {
    intp numLabels = labelMatrix_->numLabels_;
    intp offset = statisticIndex * numLabels;

    for (intp c = 0; c < numPredictions_; c++) {
        intp l = labelIndices_ != NULL ? labelIndices_[c] : c;

        // Only uncovered labels must be considered...
        if (uncoveredLabels_[offset + l] > 0) {
            // Add the current example and label to the confusion matrix for the current label...
            uint8 trueLabel = labelMatrix_->getLabel(statisticIndex, l);
            uint8 predictedLabel = minorityLabels_[l];
            intp element = getConfusionMatrixElement(trueLabel, predictedLabel);
            confusionMatricesCovered_[c * 4 + element] += weight;
        }
    }
}

void LabelWiseRefinementSearchImpl::resetSearch() {
    // Allocate an array for storing the accumulated confusion matrices, if necessary...
    if (accumulatedConfusionMatricesCovered_ == NULL) {
        accumulatedConfusionMatricesCovered_ = (float64*) malloc(numPredictions_ * 4 * sizeof(float64));
        arrays::setToZeros(accumulatedConfusionMatricesCovered_, numPredictions_, 4);
    }

    // Reset the confusion matrix for each label to zero and add its elements to the accumulated confusion matrix...
    for (intp c = 0; c < numPredictions_; c++) {
        intp offset = c * 4;

        for (intp i = 0; i < 4; i++) {
            intp j = offset + i;
            accumulatedConfusionMatricesCovered_[j] += confusionMatricesCovered_[j];
            confusionMatricesCovered_[j] = 0;
        }
    }
}

LabelWisePrediction* LabelWiseRefinementSearchImpl::calculateLabelWisePrediction(bool uncovered,
                                                                                                  bool accumulated) {
    float64* confusionMatricesCovered = accumulated ? accumulatedConfusionMatricesCovered_ : confusionMatricesCovered_;
    ruleEvaluation_->calculateLabelWisePrediction(labelIndices_, minorityLabels_, confusionMatricesTotal_,
                                                  confusionMatricesSubset_, confusionMatricesCovered, uncovered,
                                                  prediction_);
    return prediction_;
}

LabelWiseStatisticsImpl::LabelWiseStatisticsImpl(LabelWiseRuleEvaluationImpl* ruleEvaluation) {
    ruleEvaluation_ = ruleEvaluation;
    // TODO
}

LabelWiseStatisticsImpl::~LabelWiseStatisticsImpl() {
    // TODO
}

void LabelWiseStatisticsImpl::applyDefaultPrediction(AbstractLabelMatrix* labelMatrix,
                                                     DefaultPrediction* defaultPrediction) {
    // TODO
}

void LabelWiseStatisticsImpl::resetSampledStatistics() {
    // TODO
}

void LabelWiseStatisticsImpl::addSampledStatistic(intp statisticIndex, uint32 weight) {
    // TODO
}

void LabelWiseStatisticsImpl::resetCoveredStatistics() {
    // TODO
}

void LabelWiseStatisticsImpl::updateCoveredStatistic(intp statisticIndex, uint32 weight, bool remove) {
    // TODO
}

AbstractRefinementSearch* LabelWiseStatisticsImpl::beginSearch(const intp* labelIndices) {
    // TODO
    return NULL;
}

void LabelWiseStatisticsImpl::applyPrediction(intp statisticIndex, const intp* labelIndices, HeadCandidate* head) {
    // TODO
}
