#include "index_vector_partial.h"
#include "../head_refinement/head_refinement.h"
#include "../head_refinement/head_refinement_factory.h"
#include "../statistics.h"
#include "../thresholds.h"


PartialIndexVector::PartialIndexVector(uint32 numElements)
    : vector_(DenseVector<uint32>(numElements)) {

}

bool PartialIndexVector::isPartial() const {
    return true;
}

uint32 PartialIndexVector::getNumElements() const {
    return vector_.getNumElements();
}

void PartialIndexVector::setNumElements(uint32 numElements) {
    vector_.setNumElements(numElements);
}

uint32 PartialIndexVector::getIndex(uint32 pos) const {
    return vector_.getValue(pos);
}

PartialIndexVector::iterator PartialIndexVector::begin() {
    return vector_.begin();
}

PartialIndexVector::iterator PartialIndexVector::end() {
    return vector_.end();
}

PartialIndexVector::const_iterator PartialIndexVector::cbegin() const {
    return vector_.cbegin();
}

PartialIndexVector::const_iterator PartialIndexVector::cend() const {
    return vector_.cend();
}

std::unique_ptr<IStatisticsSubset> PartialIndexVector::createSubset(const IHistogram& histogram) const {
    return histogram.createSubset(*this);
}

std::unique_ptr<IRuleRefinement> PartialIndexVector::createRuleRefinement(IThresholdsSubset& thresholdsSubset,
                                                                          uint32 featureIndex) const {
    return thresholdsSubset.createRuleRefinement(*this, featureIndex);
}

std::unique_ptr<IHeadRefinement> PartialIndexVector::createHeadRefinement(const IHeadRefinementFactory& factory) const {
    return factory.create(*this);
}
