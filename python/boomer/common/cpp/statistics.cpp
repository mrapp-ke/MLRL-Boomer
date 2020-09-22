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
