#include "statistics.h"


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

IStatisticsSubset* AbstractStatistics::createSubset(uint32 numLabelIndices, const uint32* labelIndices) {
    return NULL;
}

void AbstractStatistics::applyPrediction(uint32 statisticIndex, Prediction* prediction) {

}
