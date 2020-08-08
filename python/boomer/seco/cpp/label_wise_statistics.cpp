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
    arrays::setToZeros(confusionMatricesCovered_, numPredictions * 4);
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
        arrays::setToZeros(accumulatedConfusionMatricesCovered_, numPredictions_ * 4);
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

LabelWisePrediction* LabelWiseRefinementSearchImpl::calculateLabelWisePrediction(bool uncovered, bool accumulated) {
    float64* confusionMatricesCovered = accumulated ? accumulatedConfusionMatricesCovered_ : confusionMatricesCovered_;
    ruleEvaluation_->calculateLabelWisePrediction(labelIndices_, minorityLabels_, confusionMatricesTotal_,
                                                  confusionMatricesSubset_, confusionMatricesCovered, uncovered,
                                                  prediction_);
    return prediction_;
}

LabelWiseStatisticsImpl::LabelWiseStatisticsImpl(LabelWiseRuleEvaluationImpl* ruleEvaluation) {
    ruleEvaluation_ = ruleEvaluation;
    uncoveredLabels_ = NULL;
    minorityLabels_ = NULL;
    confusionMatricesTotal_ = NULL;
    confusionMatricesSubset_ = NULL;
}

LabelWiseStatisticsImpl::~LabelWiseStatisticsImpl() {
    free(uncoveredLabels_);
    free(minorityLabels_);
    free(confusionMatricesTotal_);
    free(confusionMatricesSubset_);
}

void LabelWiseStatisticsImpl::applyDefaultPrediction(AbstractLabelMatrix* labelMatrix,
                                                     DefaultPrediction* defaultPrediction) {
    // The number of examples
    intp numExamples = labelMatrix->numExamples_;
    // The number of labels
    intp numLabels = labelMatrix->numLabels_;
    // A matrix that stores the weights of individual examples and labels that are still uncovered
    float64* uncoveredLabels = (float64*) malloc(numExamples * numLabels * sizeof(float64));
    // The sum of weights of all examples and labels that remain to be covered
    float64 sumUncoveredLabels = 0;
    // An array that stores whether rules should predict individual labels as relevant (1) or irrelevant (0)
    uint8* minorityLabels = (uint8*) malloc(numLabels * sizeof(uint8));
    // A matrix that stores a confusion matrix, which takes into account all examples, for each label
    float64* confusionMatricesTotal = (float64*) malloc(numLabels * 4 * sizeof(float64));
    // A matrix that stores a confusion matrix, which takes into account the examples covered by the previous refinement
    // of a rule, for each label
    float64* confusionMatricesSubset = (float64*) malloc(numLabels * 4 * sizeof(float64));
    // An array that stores the predictions of the default rule of NULL, if no default rule is used
    float64* predictedScores = defaultPrediction == NULL ? NULL : defaultPrediction->predictedScores_;

    for (intp c = 0; c < numLabels; c++) {
        uint8 predictedLabel = predictedScores != NULL ? (uint8) predictedScores[c] : 0;

        // Rules should predict the opposite of the default rule...
        minorityLabels[c] = predictedLabel > 0 ? 0 : 1;

        for (intp r = 0; r < numExamples; r++) {
            uint8 trueLabel = labelMatrix->getLabel(r, c);

            // Increment the total number of uncovered labels, if the default rule's prediction for the current example
            // and label is incorrect...
            if (predictedLabel != trueLabel) {
                sumUncoveredLabels++;
            }

            // Mark the current example and label as uncovered...
            uncoveredLabels[r * numLabels + c] = 1;
        }
    }

    // Store class members...
    labelMatrix_ = labelMatrix;
    uncoveredLabels_ = uncoveredLabels;
    sumUncoveredLabels_ = sumUncoveredLabels;
    minorityLabels_ = minorityLabels;
    confusionMatricesTotal_ = confusionMatricesTotal;
    confusionMatricesSubset_ = confusionMatricesSubset;
}

void LabelWiseStatisticsImpl::resetSampledStatistics() {
    intp numLabels = labelMatrix_->numLabels_;
    intp numElements = numLabels * 4;
    arrays::setToZeros(confusionMatricesTotal_, numElements);
    arrays::setToZeros(confusionMatricesSubset_, numElements);
}

void LabelWiseStatisticsImpl::addSampledStatistic(intp statisticIndex, uint32 weight) {
    intp numLabels = labelMatrix_->numLabels_;
    intp offset = statisticIndex * numLabels;

    for (intp c = 0; c < numLabels; c++) {
        float64 labelWeight = uncoveredLabels_[offset + c];

        // Only uncovered labels must be considered...
        if (labelWeight > 0) {
            // Add the current example and label to the confusion matrix that corresponds to the current label...
            uint8 trueLabel = labelMatrix_->getLabel(statisticIndex, c);
            uint8 predictedLabel = minorityLabels_[c];
            intp element = getConfusionMatrixElement(trueLabel, predictedLabel);
            confusionMatricesTotal_[c * 4 + element] += weight;
            confusionMatricesSubset_[c * 4 + element] += weight;
        }
    }
}

void LabelWiseStatisticsImpl::resetCoveredStatistics() {
    // Reset confusion matrices to 0...
    intp numLabels = labelMatrix_->numLabels_;
    int numElements = numLabels * 4;
    arrays::setToZeros(confusionMatricesSubset_, numElements);
}

void LabelWiseStatisticsImpl::updateCoveredStatistic(intp statisticIndex, uint32 weight, bool remove) {
    intp numLabels = labelMatrix_->numLabels_;
    intp offset = statisticIndex * numLabels;
    float64 signedWeight = remove ? -((float64) weight) : weight;

    for (intp c = 0; c < numLabels; c++) {
        float64 labelWeight = uncoveredLabels_[offset + c];

        // Only uncovered labels must be considered...
        if (labelWeight > 0) {
            // Add the current example and label to the confusion matrix that corresponds to the current label...
            uint8 trueLabel = labelMatrix_->getLabel(statisticIndex, c);
            uint8 predictedLabel = minorityLabels_[c];
            intp element = getConfusionMatrixElement(trueLabel, predictedLabel);
            confusionMatricesSubset_[c * 4 + element] += signedWeight;
        }
    }
}

AbstractRefinementSearch* LabelWiseStatisticsImpl::beginSearch(intp numLabelIndices, const intp* labelIndices) {
    intp numPredictions = labelIndices == NULL ? labelMatrix_->numLabels_ : numLabelIndices;
    return new LabelWiseRefinementSearchImpl(ruleEvaluation_, numPredictions, labelIndices, labelMatrix_,
                                             uncoveredLabels_, minorityLabels_, confusionMatricesTotal_,
                                             confusionMatricesSubset_);
}

void LabelWiseStatisticsImpl::applyPrediction(intp statisticIndex, const intp* labelIndices, HeadCandidate* head) {
    intp numPredictions = head->numPredictions_;
    intp numLabels = labelMatrix_->numLabels_;
    intp offset = statisticIndex * numLabels;

    // Only the labels that are predicted by the new rule must be considered...
    for (intp c = 0; c < numPredictions; c++) {
        intp l = labelIndices != NULL ? labelIndices[c] : c;
        intp i = offset + l;
        float64 labelWeight = uncoveredLabels_[i];

        if (labelWeight > 0) {
            uint8 trueLabel = labelMatrix_->getLabel(statisticIndex, l);
            uint8 predictedLabel = minorityLabels_[l];

            // Decrement the total sum of uncovered labels, if the default rule's prediction for the current example and
            // label is incorrect...
            if (predictedLabel != trueLabel) {
                sumUncoveredLabels_ -= labelWeight;
            }

            // Mark the current example and label as covered...
            uncoveredLabels_[i] = 0;
        }
    }
}
