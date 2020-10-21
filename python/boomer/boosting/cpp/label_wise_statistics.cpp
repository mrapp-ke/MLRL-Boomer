#include "label_wise_statistics.h"
#include <stdlib.h>
#include <cstddef>

using namespace boosting;


AbstractLabelWiseStatistics::AbstractLabelWiseStatistics(
        uint32 numStatistics, uint32 numLabels, std::shared_ptr<ILabelWiseRuleEvaluation> ruleEvaluationPtr)
    : AbstractGradientStatistics(numStatistics, numLabels), ruleEvaluationPtr_(ruleEvaluationPtr) {

}

void AbstractLabelWiseStatistics::setRuleEvaluation(std::shared_ptr<ILabelWiseRuleEvaluation> ruleEvaluationPtr) {
    ruleEvaluationPtr_ = ruleEvaluationPtr;
}

DenseLabelWiseStatisticsImpl::StatisticsSubsetImpl::StatisticsSubsetImpl(const DenseLabelWiseStatisticsImpl& statistics,
                                                                         uint32 numPredictions,
                                                                         const uint32* labelIndices)
    : statistics_(statistics), numPredictions_(numPredictions), labelIndices_(labelIndices),
      prediction_(LabelWiseEvaluatedPrediction(numPredictions)) {
    sumsOfGradients_ = (float64*) malloc(numPredictions * sizeof(float64));
    arrays::setToZeros(sumsOfGradients_, numPredictions);
    accumulatedSumsOfGradients_ = nullptr;
    sumsOfHessians_ = (float64*) malloc(numPredictions * sizeof(float64));
    arrays::setToZeros(sumsOfHessians_, numPredictions);
    accumulatedSumsOfHessians_ = nullptr;
}

DenseLabelWiseStatisticsImpl::StatisticsSubsetImpl::~StatisticsSubsetImpl() {
    free(sumsOfGradients_);
    free(accumulatedSumsOfGradients_);
    free(sumsOfHessians_);
    free(accumulatedSumsOfHessians_);
}

void DenseLabelWiseStatisticsImpl::StatisticsSubsetImpl::addToSubset(uint32 statisticIndex, uint32 weight) {
    // For each label, add the gradient and Hessian of the example at the given index (weighted by the given weight) to
    // the current sum of gradients and Hessians...
    uint32 offset = statisticIndex * statistics_.getNumCols();

    for (uint32 c = 0; c < numPredictions_; c++) {
        uint32 l = labelIndices_ != nullptr ? labelIndices_[c] : c;
        uint32 i = offset + l;
        sumsOfGradients_[c] += (weight * statistics_.gradients_[i]);
        sumsOfHessians_[c] += (weight * statistics_.hessians_[i]) ;
    }
}

void DenseLabelWiseStatisticsImpl::StatisticsSubsetImpl::resetSubset() {
    // Allocate arrays for storing the accumulated sums of gradients and Hessians, if necessary...
    if (accumulatedSumsOfGradients_ == nullptr) {
        accumulatedSumsOfGradients_ = (float64*) malloc(numPredictions_ * sizeof(float64));
        arrays::setToZeros(accumulatedSumsOfGradients_, numPredictions_);
        accumulatedSumsOfHessians_ = (float64*) malloc(numPredictions_ * sizeof(float64));
        arrays::setToZeros(accumulatedSumsOfHessians_, numPredictions_);
    }

    // Reset the sum of gradients and Hessians for each label to zero and add it to the accumulated sums of gradients
    // and hessians...
    for (uint32 c = 0; c < numPredictions_; c++) {
        accumulatedSumsOfGradients_[c] += sumsOfGradients_[c];
        sumsOfGradients_[c] = 0;
        accumulatedSumsOfHessians_[c] += sumsOfHessians_[c];
        sumsOfHessians_[c] = 0;
    }
}

const LabelWiseEvaluatedPrediction& DenseLabelWiseStatisticsImpl::StatisticsSubsetImpl::calculateLabelWisePrediction(
        bool uncovered, bool accumulated) {
    float64* sumsOfGradients = accumulated ? accumulatedSumsOfGradients_ : sumsOfGradients_;
    float64* sumsOfHessians = accumulated ? accumulatedSumsOfHessians_ : sumsOfHessians_;
    statistics_.ruleEvaluationPtr_->calculateLabelWisePrediction(labelIndices_, statistics_.totalSumsOfGradients_,
                                                                 sumsOfGradients, statistics_.totalSumsOfHessians_,
                                                                 sumsOfHessians, uncovered, prediction_);
    return prediction_;
}

DenseLabelWiseStatisticsImpl::HistogramBuilderImpl::HistogramBuilderImpl(const DenseLabelWiseStatisticsImpl& statistics,
                                                                         uint32 numBins)
    : statistics_(statistics), numBins_(numBins) {
    uint32 numLabels = numBins_ * statistics.getNumCols();
    gradients_ = (float64*) calloc(numLabels, sizeof(float64));
    hessians_ = (float64*) calloc(numLabels, sizeof(float64));
}

void DenseLabelWiseStatisticsImpl::HistogramBuilderImpl::onBinUpdate(uint32 binIndex,
                                                                     const FeatureVector::Entry& entry) {
    uint32 numLabels = statistics_.getNumCols();
    uint32 index = entry.index;
    uint32 offset = index * numLabels;
    uint32 binOffset = binIndex * numLabels;

    for(uint32 c = 0; c < numLabels; c++) {
        float64 gradient = statistics_.gradients_[offset + c];
        float64 hessian = statistics_.hessians_[offset + c];
        gradients_[binOffset + c] += gradient;
        hessians_[binOffset + c] += hessian;
    }
}

std::unique_ptr<AbstractStatistics> DenseLabelWiseStatisticsImpl::HistogramBuilderImpl::build() const {
    return std::make_unique<DenseLabelWiseStatisticsImpl>(statistics_.lossFunctionPtr_, statistics_.ruleEvaluationPtr_,
                                                          statistics_.labelMatrixPtr_, gradients_, hessians_,
                                                          statistics_.currentScores_);
}

DenseLabelWiseStatisticsImpl::DenseLabelWiseStatisticsImpl(std::shared_ptr<ILabelWiseLoss> lossFunctionPtr,
                                                           std::shared_ptr<ILabelWiseRuleEvaluation> ruleEvaluationPtr,
                                                           std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr,
                                                           float64* gradients, float64* hessians,
                                                           float64* currentScores)
    : AbstractLabelWiseStatistics(labelMatrixPtr->getNumRows(), labelMatrixPtr->getNumCols(), ruleEvaluationPtr),
      lossFunctionPtr_(lossFunctionPtr), labelMatrixPtr_(labelMatrixPtr), gradients_(gradients), hessians_(hessians),
      currentScores_(currentScores) {
    // The number of labels
    uint32 numLabels = this->getNumCols();
    // An array that stores the column-wise sums of the matrix of gradients
    totalSumsOfGradients_ = (float64*) malloc(numLabels * sizeof(float64));
    // An array that stores the column-wise sums of the matrix of hessians
    totalSumsOfHessians_ = (float64*) malloc(numLabels * sizeof(float64));
}

DenseLabelWiseStatisticsImpl::~DenseLabelWiseStatisticsImpl() {
    free(currentScores_);
    free(gradients_);
    free(totalSumsOfGradients_);
    free(hessians_);
    free(totalSumsOfHessians_);
}

void DenseLabelWiseStatisticsImpl::resetCoveredStatistics() {
    uint32 numLabels = this->getNumCols();
    arrays::setToZeros(totalSumsOfGradients_, numLabels);
    arrays::setToZeros(totalSumsOfHessians_, numLabels);
}

void DenseLabelWiseStatisticsImpl::updateCoveredStatistic(uint32 statisticIndex, uint32 weight, bool remove) {
    uint32 numLabels = this->getNumCols();
    uint32 offset = statisticIndex * numLabels;
    float64 signedWeight = remove ? -((float64) weight) : weight;

    // For each label, add the gradient and Hessian of the example at the given index (weighted by the given weight) to
    // the total sums of gradients and Hessians...
    for (uint32 c = 0; c < numLabels; c++) {
        uint32 i = offset + c;
        totalSumsOfGradients_[c] += (signedWeight * gradients_[i]);
        totalSumsOfHessians_[c] += (signedWeight * hessians_[i]);
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
        float64 predictedScore = predictedScores[c];
        uint32 i = offset + l;

        // Update the score that is currently predicted for the current example and label...
        float64 updatedScore = currentScores_[i] + predictedScore;
        currentScores_[i] = updatedScore;

        // Update the gradient and Hessian for the current example and label...
        std::pair<float64, float64> pair = lossFunctionPtr_->calculateGradientAndHessian(*labelMatrixPtr_,
                                                                                         statisticIndex, l,
                                                                                         updatedScore);
        gradients_[i] = pair.first;
        hessians_[i] = pair.second;
    }
}

std::unique_ptr<AbstractStatistics::IHistogramBuilder> DenseLabelWiseStatisticsImpl::buildHistogram(
        uint32 numBins) const {
    return std::make_unique<DenseLabelWiseStatisticsImpl::HistogramBuilderImpl>(*this, numBins);
}

DenseLabelWiseStatisticsFactoryImpl::DenseLabelWiseStatisticsFactoryImpl(
        std::shared_ptr<ILabelWiseLoss> lossFunctionPtr, std::shared_ptr<ILabelWiseRuleEvaluation> ruleEvaluationPtr,
        std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr) {
    lossFunctionPtr_ = lossFunctionPtr;
    ruleEvaluationPtr_ = ruleEvaluationPtr;
    labelMatrixPtr_ = labelMatrixPtr;
}

std::unique_ptr<AbstractLabelWiseStatistics> DenseLabelWiseStatisticsFactoryImpl::create() const {
    // The number of examples
    uint32 numExamples = labelMatrixPtr_->getNumRows();
    // The number of labels
    uint32 numLabels = labelMatrixPtr_->getNumCols();
    // A matrix that stores the gradients for each example and label
    float64* gradients = (float64*) malloc(numExamples * numLabels * sizeof(float64));
    // A matrix that stores the Hessians for each example and label
    float64* hessians = (float64*) malloc(numExamples * numLabels * sizeof(float64));
    // A matrix that stores the currently predicted scores for each example and label
    float64* currentScores = (float64*) malloc(numExamples * numLabels * sizeof(float64));

    for (uint32 r = 0; r < numExamples; r++) {
        uint32 offset = r * numLabels;

        for (uint32 c = 0; c < numLabels; c++) {
            uint32 i = offset + c;

            // Calculate the initial gradient and Hessian for the current example and label...
            std::pair<float64, float64> pair = lossFunctionPtr_->calculateGradientAndHessian(*labelMatrixPtr_, r, c, 0);
            gradients[i] = pair.first;
            hessians[i] = pair.second;

            // Store the score that is initially predicted for the current example and label...
            currentScores[i] = 0;
        }
    }

    return std::make_unique<DenseLabelWiseStatisticsImpl>(lossFunctionPtr_, ruleEvaluationPtr_, labelMatrixPtr_,
                                                          gradients, hessians, currentScores);
}
