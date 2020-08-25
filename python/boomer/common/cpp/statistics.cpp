#include "statistics.h"


AbstractRefinementSearch::~AbstractRefinementSearch() {

}

void AbstractRefinementSearch::updateSearch(intp statisticIndex, uint32 weight) {

}

void AbstractRefinementSearch::resetSearch() {

}

LabelWisePredictionCandidate* AbstractRefinementSearch::calculateLabelWisePrediction(bool uncovered, bool accumulated) {
    return NULL;
}

PredictionCandidate* AbstractRefinementSearch::calculateExampleWisePrediction(bool uncovered, bool accumulated) {
    return NULL;
}

PredictionCandidate* AbstractDecomposableRefinementSearch::calculateExampleWisePrediction(bool uncovered,
                                                                                          bool accumulated) {
    // In the decomposable case, the example-wise predictions are the same as the label-wise predictions...
    return (PredictionCandidate*) this->calculateLabelWisePrediction(uncovered, accumulated);
}

AbstractStatistics::~AbstractStatistics() {

}

void AbstractStatistics::applyDefaultPrediction(std::shared_ptr<AbstractLabelMatrix> labelMatrixPtr,
                                                Prediction* defaultPrediction) {

}

void AbstractStatistics::resetSampledStatistics() {

}

void AbstractStatistics::addSampledStatistic(intp statisticIndex, uint32 weight) {

}

void AbstractStatistics::resetCoveredStatistics() {

}

void AbstractStatistics::updateCoveredStatistic(intp statisticIndex, uint32 weight, bool remove) {

}

AbstractRefinementSearch* AbstractStatistics::beginSearch(intp numLabelIndices, const intp* labelIndices) {
    return NULL;
}

void AbstractStatistics::applyPrediction(intp statisticIndex, HeadCandidate* head) {

}
