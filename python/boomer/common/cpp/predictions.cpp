#include "predictions.h"
#include "statistics.h"
#include "thresholds.h"
#include <cstdlib>


AbstractPrediction::AbstractPrediction(uint32 numElements)
    : DenseVector<float64>(numElements) {

}

AbstractEvaluatedPrediction::AbstractEvaluatedPrediction(uint32 numElements)
    : AbstractPrediction(numElements) {

}

FullPrediction::FullPrediction(uint32 numElements)
    : AbstractEvaluatedPrediction(numElements), indexVector_(FullIndexVector(numElements)) {

}

FullPrediction::index_const_iterator FullPrediction::indices_cbegin() const {
    return indexVector_.indices_cbegin();
}

FullPrediction::index_const_iterator FullPrediction::indices_cend() const {
    return indexVector_.indices_cend();
}

uint32 FullPrediction::getNumElements() const {
    return DenseVector<float64>::getNumElements();
}

void FullPrediction::setNumElements(uint32 numElements) {
    DenseVector<float64>::setNumElements(numElements);
    indexVector_.setNumElements(numElements);
}

bool FullPrediction::isPartial() const {
    return false;
}

uint32 FullPrediction::getIndex(uint32 pos) const {
    return indexVector_.getIndex(pos);
}

std::unique_ptr<IStatisticsSubset> FullPrediction::createSubset(const AbstractStatistics& statistics) const {
    return indexVector_.createSubset(statistics);
}

std::unique_ptr<IRuleRefinement> FullPrediction::createRuleRefinement(IThresholdsSubset& thresholdsSubset,
                                                                         uint32 featureIndex) const {
    return indexVector_.createRuleRefinement(thresholdsSubset, featureIndex);
}

std::unique_ptr<IHeadRefinement> FullPrediction::createHeadRefinement(const IHeadRefinementFactory& factory) const {
    return indexVector_.createHeadRefinement(factory);
}

void FullPrediction::apply(AbstractStatistics& statistics, uint32 statisticIndex) const {
    statistics.applyPrediction(statisticIndex, *this);
}

PartialPrediction::PartialPrediction(uint32 numElements)
    : AbstractEvaluatedPrediction(numElements), indexVector_(PartialIndexVector(numElements)) {

}

PartialPrediction::index_iterator PartialPrediction::indices_begin() {
    return indexVector_.indices_begin();
}

PartialPrediction::index_iterator PartialPrediction::indices_end() {
    return indexVector_.indices_end();
}

PartialPrediction::index_const_iterator PartialPrediction::indices_cbegin() const {
    return indexVector_.indices_cbegin();
}

PartialPrediction::index_const_iterator PartialPrediction::indices_cend() const {
    return indexVector_.indices_cend();
}

uint32 PartialPrediction::getNumElements() const {
    return DenseVector<float64>::getNumElements();
}

void PartialPrediction::setNumElements(uint32 numElements) {
    DenseVector<float64>::setNumElements(numElements);
    indexVector_.setNumElements(numElements);
}

bool PartialPrediction::isPartial() const {
    return true;
}

uint32 PartialPrediction::getIndex(uint32 pos) const {
    return indexVector_.getIndex(pos);
}

std::unique_ptr<IStatisticsSubset> PartialPrediction::createSubset(const AbstractStatistics& statistics) const {
    return indexVector_.createSubset(statistics);
}

std::unique_ptr<IRuleRefinement> PartialPrediction::createRuleRefinement(IThresholdsSubset& thresholdsSubset,
                                                                         uint32 featureIndex) const {
    return indexVector_.createRuleRefinement(thresholdsSubset, featureIndex);
}

std::unique_ptr<IHeadRefinement> PartialPrediction::createHeadRefinement(const IHeadRefinementFactory& factory) const {
    return indexVector_.createHeadRefinement(factory);
}

void PartialPrediction::apply(AbstractStatistics& statistics, uint32 statisticIndex) const {
    statistics.applyPrediction(statisticIndex, *this);
}
