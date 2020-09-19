#include "statistics.h"


AbstractStatisticsSubset::~AbstractStatisticsSubset() {

}

void AbstractStatisticsSubset::updateSearch(uint32 statisticIndex, uint32 weight) {

}

void AbstractStatisticsSubset::resetSearch() {

}

LabelWisePredictionCandidate* AbstractStatisticsSubset::calculateLabelWisePrediction(bool uncovered, bool accumulated) {
    return NULL;
}

PredictionCandidate* AbstractStatisticsSubset::calculateExampleWisePrediction(bool uncovered, bool accumulated) {
    return NULL;
}

PredictionCandidate* AbstractDecomposableStatisticsSubset::calculateExampleWisePrediction(bool uncovered,
                                                                                          bool accumulated) {
    // In the decomposable case, the example-wise predictions are the same as the label-wise predictions...
    return (PredictionCandidate*) this->calculateLabelWisePrediction(uncovered, accumulated);
}

AbstractStatistics::AbstractStatistics(uint32 numStatistics, uint32 numLabels) {
    numStatistics_ = numStatistics;
    numLabels_ = numLabels;
}

uint32 AbstractStatistics::getNumRows() {
    return numStatistics_;
}

uint32 AbstractStatistics::getNumCols() {
    return numLabels_;
}

void AbstractStatistics::resetSampledStatistics() {

}

void AbstractStatistics::addSampledStatistic(uint32 statisticIndex, uint32 weight) {

}

void AbstractStatistics::resetCoveredStatistics() {

}

void AbstractStatistics::updateCoveredStatistic(uint32 statisticIndex, uint32 weight, bool remove) {

}

AbstractStatisticsSubset* AbstractStatistics::beginSearch(uint32 numLabelIndices, const uint32* labelIndices) {
    return NULL;
}

void AbstractStatistics::applyPrediction(uint32 statisticIndex, Prediction* prediction) {

}
