#include "label_wise_statistics.h"
#include "heuristics.h"
#include <stdlib.h>
#include <cstddef>

using namespace seco;


AbstractLabelWiseStatistics::AbstractLabelWiseStatistics(uint32 numStatistics, uint32 numLabels,
                                                         float64 sumUncoveredLabels,
                                                         std::shared_ptr<ILabelWiseRuleEvaluation> ruleEvaluationPtr)
    : AbstractCoverageStatistics(numStatistics, numLabels, sumUncoveredLabels), ruleEvaluationPtr_(ruleEvaluationPtr) {

}

void AbstractLabelWiseStatistics::setRuleEvaluation(std::shared_ptr<ILabelWiseRuleEvaluation> ruleEvaluationPtr) {
    ruleEvaluationPtr_ = ruleEvaluationPtr;
}

DenseLabelWiseStatisticsImpl::StatisticsSubsetImpl::StatisticsSubsetImpl(const DenseLabelWiseStatisticsImpl& statistics,
                                                                         uint32 numPredictions,
                                                                         const uint32* labelIndices)
    : statistics_(statistics), numPredictions_(numPredictions), labelIndices_(labelIndices),
      prediction_(LabelWiseEvaluatedPrediction(numPredictions)) {
    confusionMatricesCovered_ = (float64*) malloc(numPredictions * NUM_CONFUSION_MATRIX_ELEMENTS * sizeof(float64));
    arrays::setToZeros(confusionMatricesCovered_, numPredictions * NUM_CONFUSION_MATRIX_ELEMENTS);
    accumulatedConfusionMatricesCovered_ = nullptr;
}

DenseLabelWiseStatisticsImpl::StatisticsSubsetImpl::~StatisticsSubsetImpl() {
    free(confusionMatricesCovered_);
    free(accumulatedConfusionMatricesCovered_);
}

void DenseLabelWiseStatisticsImpl::StatisticsSubsetImpl::addToSubset(uint32 statisticIndex, uint32 weight) {
    uint32 numLabels = statistics_.labelMatrixPtr_->getNumCols();
    uint32 offset = statisticIndex * numLabels;

    for (uint32 c = 0; c < numPredictions_; c++) {
        uint32 l = labelIndices_ != nullptr ? labelIndices_[c] : c;

        // Only uncovered labels must be considered...
        if (statistics_.uncoveredLabels_[offset + l] > 0) {
            // Add the current example and label to the confusion matrix for the current label...
            uint8 trueLabel = statistics_.labelMatrixPtr_->getValue(statisticIndex, l);
            uint8 predictedLabel = statistics_.minorityLabels_[l];
            uint32 element = getConfusionMatrixElement(trueLabel, predictedLabel);
            confusionMatricesCovered_[c * NUM_CONFUSION_MATRIX_ELEMENTS + element] += weight;
        }
    }
}

void DenseLabelWiseStatisticsImpl::StatisticsSubsetImpl::resetSubset() {
    // Allocate an array for storing the accumulated confusion matrices, if necessary...
    if (accumulatedConfusionMatricesCovered_ == nullptr) {
        accumulatedConfusionMatricesCovered_ =
            (float64*) malloc(numPredictions_ * NUM_CONFUSION_MATRIX_ELEMENTS * sizeof(float64));
        arrays::setToZeros(accumulatedConfusionMatricesCovered_, numPredictions_ * NUM_CONFUSION_MATRIX_ELEMENTS);
    }

    // Reset the confusion matrix for each label to zero and add its elements to the accumulated confusion matrix...
    for (uint32 c = 0; c < numPredictions_; c++) {
        uint32 offset = c * NUM_CONFUSION_MATRIX_ELEMENTS;

        for (uint32 i = 0; i < NUM_CONFUSION_MATRIX_ELEMENTS; i++) {
            uint32 j = offset + i;
            accumulatedConfusionMatricesCovered_[j] += confusionMatricesCovered_[j];
            confusionMatricesCovered_[j] = 0;
        }
    }
}

const LabelWiseEvaluatedPrediction& DenseLabelWiseStatisticsImpl::StatisticsSubsetImpl::calculateLabelWisePrediction(
        bool uncovered, bool accumulated) {
    float64* confusionMatricesCovered = accumulated ? accumulatedConfusionMatricesCovered_ : confusionMatricesCovered_;
    statistics_.ruleEvaluationPtr_->calculateLabelWisePrediction(labelIndices_, statistics_.minorityLabels_,
                                                                 statistics_.confusionMatricesTotal_,
                                                                 statistics_.confusionMatricesSubset_,
                                                                 confusionMatricesCovered, uncovered, prediction_);
    return prediction_;
}

DenseLabelWiseStatisticsImpl::DenseLabelWiseStatisticsImpl(std::shared_ptr<ILabelWiseRuleEvaluation> ruleEvaluationPtr,
                                                           std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr,
                                                           float64* uncoveredLabels, float64 sumUncoveredLabels,
                                                           uint8* minorityLabels)
    : AbstractLabelWiseStatistics(labelMatrixPtr->getNumRows(), labelMatrixPtr->getNumCols(), sumUncoveredLabels,
                                  ruleEvaluationPtr),
      labelMatrixPtr_(labelMatrixPtr), uncoveredLabels_(uncoveredLabels), minorityLabels_(minorityLabels) {
    // The number of labels
    uint32 numLabels = this->getNumCols();
    // A matrix that stores a confusion matrix, which takes into account all examples, for each label
    confusionMatricesTotal_ = (float64*) malloc(numLabels * NUM_CONFUSION_MATRIX_ELEMENTS * sizeof(float64));
    // A matrix that stores a confusion matrix, which takes into account the examples covered by the previous refinement
    // of a rule, for each label
    confusionMatricesSubset_ = (float64*) malloc(numLabels * NUM_CONFUSION_MATRIX_ELEMENTS * sizeof(float64));
}

DenseLabelWiseStatisticsImpl::~DenseLabelWiseStatisticsImpl() {
    free(uncoveredLabels_);
    free(minorityLabels_);
    free(confusionMatricesTotal_);
    free(confusionMatricesSubset_);
}

void DenseLabelWiseStatisticsImpl::resetSampledStatistics() {
    uint32 numLabels = this->getNumCols();
    uint32 numElements = numLabels * NUM_CONFUSION_MATRIX_ELEMENTS;
    arrays::setToZeros(confusionMatricesTotal_, numElements);
    arrays::setToZeros(confusionMatricesSubset_, numElements);
}

void DenseLabelWiseStatisticsImpl::addSampledStatistic(uint32 statisticIndex, uint32 weight) {
    uint32 numLabels = this->getNumCols();
    uint32 offset = statisticIndex * numLabels;

    for (uint32 c = 0; c < numLabels; c++) {
        float64 labelWeight = uncoveredLabels_[offset + c];

        // Only uncovered labels must be considered...
        if (labelWeight > 0) {
            // Add the current example and label to the confusion matrix that corresponds to the current label...
            uint8 trueLabel = labelMatrixPtr_->getValue(statisticIndex, c);
            uint8 predictedLabel = minorityLabels_[c];
            uint32 element = getConfusionMatrixElement(trueLabel, predictedLabel);
            uint32 i = c * NUM_CONFUSION_MATRIX_ELEMENTS + element;
            confusionMatricesTotal_[i] += weight;
            confusionMatricesSubset_[i] += weight;
        }
    }
}

void DenseLabelWiseStatisticsImpl::resetCoveredStatistics() {
    // Reset confusion matrices to 0...
    uint32 numLabels = this->getNumCols();
    uint32 numElements = numLabels * NUM_CONFUSION_MATRIX_ELEMENTS;
    arrays::setToZeros(confusionMatricesSubset_, numElements);
}

void DenseLabelWiseStatisticsImpl::updateCoveredStatistic(uint32 statisticIndex, uint32 weight, bool remove) {
    uint32 numLabels = this->getNumCols();
    uint32 offset = statisticIndex * numLabels;
    float64 signedWeight = remove ? -((float64) weight) : weight;

    for (uint32 c = 0; c < numLabels; c++) {
        float64 labelWeight = uncoveredLabels_[offset + c];

        // Only uncovered labels must be considered...
        if (labelWeight > 0) {
            // Add the current example and label to the confusion matrix that corresponds to the current label...
            uint8 trueLabel = labelMatrixPtr_->getValue(statisticIndex, c);
            uint8 predictedLabel = minorityLabels_[c];
            uint32 element = getConfusionMatrixElement(trueLabel, predictedLabel);
            confusionMatricesSubset_[c * NUM_CONFUSION_MATRIX_ELEMENTS + element] += signedWeight;
        }
    }
}

std::unique_ptr<IStatisticsSubset> DenseLabelWiseStatisticsImpl::createSubset(uint32 numLabelIndices,
                                                                              const uint32* labelIndices) const {
    uint32 numLabels = this->getNumCols();
    uint32 numPredictions = labelIndices == nullptr ? numLabels : numLabelIndices;
    return std::make_unique<DenseLabelWiseStatisticsImpl::StatisticsSubsetImpl>(*this, numPredictions, labelIndices);
}

void DenseLabelWiseStatisticsImpl::applyPrediction(uint32 statisticIndex, const Prediction& prediction) {
    uint32 numLabels = this->getNumCols();
    uint32 numPredictions = prediction.numPredictions_;
    const uint32* labelIndices = prediction.labelIndices_;
    const float64* predictedScores = prediction.predictedScores_;
    uint32 offset = statisticIndex * numLabels;

    // Only the labels that are predicted by the new rule must be considered...
    for (uint32 c = 0; c < numPredictions; c++) {
        uint32 l = labelIndices != nullptr ? labelIndices[c] : c;
        uint8 predictedLabel = predictedScores[c];
        uint8 minorityLabel = minorityLabels_[l];

        // Do only consider predictions that are different from the default rule's predictions...
        if (predictedLabel == minorityLabel) {
            uint32 i = offset + l;
            float64 labelWeight = uncoveredLabels_[i];

            if (labelWeight > 0) {
                uint8 trueLabel = labelMatrixPtr_->getValue(statisticIndex, l);

                // Decrement the total sum of uncovered labels, if the prediction for the current example and label is
                // correct...
                if (predictedLabel == trueLabel) {
                    sumUncoveredLabels_ -= labelWeight;
                }

                // Mark the current example and label as covered...
                uncoveredLabels_[i] = 0;
            }
        }
    }
}

std::unique_ptr<AbstractStatistics::IHistogramBuilder> DenseLabelWiseStatisticsImpl::buildHistogram(
        uint32 numBins) const {
    //TODO Support creation of histograms
    std::unique_ptr<DenseLabelWiseStatisticsImpl::IHistogramBuilder> result;
    return result;
}

DenseLabelWiseStatisticsFactoryImpl::DenseLabelWiseStatisticsFactoryImpl(
        std::shared_ptr<ILabelWiseRuleEvaluation> ruleEvaluationPtr,
        std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr)
    : ruleEvaluationPtr_(ruleEvaluationPtr), labelMatrixPtr_(labelMatrixPtr) {

}

std::unique_ptr<AbstractLabelWiseStatistics> DenseLabelWiseStatisticsFactoryImpl::create() const {
    // The number of examples
    uint32 numExamples = labelMatrixPtr_->getNumRows();
    // The number of labels
    uint32 numLabels = labelMatrixPtr_->getNumCols();
    // A matrix that stores the weights of individual examples and labels that are still uncovered
    float64* uncoveredLabels = (float64*) malloc(numExamples * numLabels * sizeof(float64));
    // The sum of weights of all examples and labels that remain to be covered
    float64 sumUncoveredLabels = 0;
    // An array that stores whether rules should predict individual labels as relevant (1) or irrelevant (0)
    uint8* minorityLabels = (uint8*) malloc(numLabels * sizeof(uint8));
    // The number of positive examples that must be exceeded for the default rule to predict a label as relevant
    float64 threshold = numExamples / 2.0;

    for (uint32 c = 0; c < numLabels; c++) {
        uint32 numPositiveLabels = 0;

        for (uint32 r = 0; r < numExamples; r++) {
            uint8 trueLabel = labelMatrixPtr_->getValue(r, c);
            numPositiveLabels += trueLabel;

            // Mark the current example and label as uncovered...
            uncoveredLabels[r * numLabels + c] = 1;
        }

        if (numPositiveLabels > threshold) {
            minorityLabels[c] = 0;
            sumUncoveredLabels += (numExamples - numPositiveLabels);
        } else {
            minorityLabels[c] = 1;
            sumUncoveredLabels += numPositiveLabels;
        }
    }

    return std::make_unique<DenseLabelWiseStatisticsImpl>(ruleEvaluationPtr_, labelMatrixPtr_, uncoveredLabels,
                                                          sumUncoveredLabels, minorityLabels);
}
