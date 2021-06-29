#include "common/head_refinement/prediction_complete.hpp"
#include "common/head_refinement/head_refinement.hpp"
#include "common/rule_refinement/rule_refinement.hpp"
#include "common/statistics/statistics.hpp"
#include "common/model/head_full.hpp"


CompletePrediction::CompletePrediction(uint32 numElements)
    : AbstractEvaluatedPrediction(numElements), indexVector_(CompleteIndexVector(numElements)) {

}

CompletePrediction::index_const_iterator CompletePrediction::indices_cbegin() const {
    return indexVector_.cbegin();
}

CompletePrediction::index_const_iterator CompletePrediction::indices_cend() const {
    return indexVector_.cend();
}

void CompletePrediction::setNumElements(uint32 numElements, bool freeMemory) {
    AbstractPrediction::setNumElements(numElements, freeMemory);
    indexVector_.setNumElements(numElements, freeMemory);
}

bool CompletePrediction::isPartial() const {
    return false;
}

uint32 CompletePrediction::getIndex(uint32 pos) const {
    return indexVector_.getIndex(pos);
}

std::unique_ptr<IStatisticsSubset> CompletePrediction::createSubset(const IImmutableStatistics& statistics) const {
    return indexVector_.createSubset(statistics);
}

std::unique_ptr<IRuleRefinement> CompletePrediction::createRuleRefinement(IThresholdsSubset& thresholdsSubset,
                                                                          uint32 featureIndex) const {
    return indexVector_.createRuleRefinement(thresholdsSubset, featureIndex);
}

std::unique_ptr<IHeadRefinement> CompletePrediction::createHeadRefinement(const IHeadRefinementFactory& factory) const {
    return indexVector_.createHeadRefinement(factory);
}

void CompletePrediction::apply(IStatistics& statistics, uint32 statisticIndex) const {
    statistics.applyPrediction(statisticIndex, *this);
}

std::unique_ptr<IHead> CompletePrediction::toHead() const {
    return std::make_unique<FullHead>(*this);
}
