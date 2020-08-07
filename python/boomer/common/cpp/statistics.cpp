#include "statistics.h"


AbstractRefinementSearch::~AbstractRefinementSearch() {

}

void AbstractRefinementSearch::updateSearch(intp statisticIndex, uint32 weight) {

}

void AbstractRefinementSearch::resetSearch() {

}

LabelWisePrediction* AbstractRefinementSearch::calculateLabelWisePrediction(bool uncovered, bool accumulated) {
    return NULL;
}

Prediction* AbstractRefinementSearch::calculateExampleWisePrediction(bool uncovered, bool accumulated) {
    return NULL;
}

Prediction* AbstractDecomposableRefinementSearch::calculateExampleWisePrediction(bool uncovered, bool accumulated) {
    // In the decomposable case, the example-wise predictions are the same as the label-wise predictions...
    return (Prediction*) this->calculateLabelWisePrediction(uncovered, accumulated);
}

AbstractStatistics::~AbstractStatistics() {

}

void AbstractStatistics::applyDefaultPrediction(AbstractLabelMatrix* labelMatrix,
                                                DefaultPrediction* defaultPrediction) {

}

void AbstractStatistics::resetSampledStatistics() {

}

void AbstractStatistics::addSampledStatistic(intp statisticIndex, uint32 weight) {

}

void AbstractStatistics::resetCoveredStatistics() {

}

void AbstractStatistics::updateCoveredStatistic(intp statisticIndex, uint32 weight, bool remove) {

}

AbstractRefinementSearch* AbstractStatistics::beginSearch(const intp* labelIndices) {

}

void AbstractStatistics::applyPrediction(intp statisticIndex, const intp* labelIndices, HeadCandidate* head) {

}
