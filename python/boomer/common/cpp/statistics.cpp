#include "statistics.h"

using namespace statistics;


AbstractRefinementSearch::~AbstractRefinementSearch() {

}

void AbstractRefinementSearch::updateSearch(intp statisticIndex, uint32 weight) {

}

void AbstractRefinementSearch::resetSearch() {

}

rule_evaluation::LabelWisePrediction* AbstractRefinementSearch::calculateLabelWisePrediction(bool uncovered,
                                                                                             bool accumulated) {
    return NULL;
}

rule_evaluation::Prediction* AbstractRefinementSearch::calculateExampleWisePrediction(bool uncovered,
                                                                                      bool accumulated) {
    return NULL;
}

rule_evaluation::Prediction* AbstractDecomposableRefinementSearch::calculateExampleWisePrediction(bool uncovered,
                                                                                                  bool accumulated) {
    // In the decomposable case, the example-wise predictions are the same as the label-wise predictions...
    return (rule_evaluation::Prediction*) this->calculateLabelWisePrediction(uncovered, accumulated);
}
