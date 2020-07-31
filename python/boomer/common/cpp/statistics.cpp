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
