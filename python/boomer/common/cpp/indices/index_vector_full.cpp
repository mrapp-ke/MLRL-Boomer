#include "index_vector_full.h"
#include "../head_refinement/head_refinement.h"
#include "../head_refinement/head_refinement_factory.h"
#include "../statistics.h"
#include "../thresholds.h"


FullIndexVector::Iterator::Iterator(uint32 index) {
    index_ = index;
}

uint32 FullIndexVector::Iterator::operator[](uint32 index) const {
    return index;
}

uint32 FullIndexVector::Iterator::operator*() const {
 return index_;
}

FullIndexVector::Iterator& FullIndexVector::Iterator::operator++(int n) {
    index_++;
    return *this;
}

bool FullIndexVector::Iterator::operator!=(const FullIndexVector::Iterator& rhs) const {
    return index_ != rhs.index_;
}

FullIndexVector::FullIndexVector(uint32 numElements) {
    numElements_ = numElements;
}

bool FullIndexVector::isPartial() const {
    return false;
}

uint32 FullIndexVector::getNumElements() const {
    return numElements_;
}

void FullIndexVector::setNumElements(uint32 numElements) {
    numElements_ = numElements;
}

uint32 FullIndexVector::getIndex(uint32 pos) const {
    return pos;
}

FullIndexVector::const_iterator FullIndexVector::cbegin() const {
    return FullIndexVector::Iterator(0);
}

FullIndexVector::const_iterator FullIndexVector::cend() const {
    return FullIndexVector::Iterator(numElements_);
}

std::unique_ptr<IStatisticsSubset> FullIndexVector::createSubset(const IHistogram& histogram) const {
    return histogram.createSubset(*this);
}

std::unique_ptr<IRuleRefinement> FullIndexVector::createRuleRefinement(IThresholdsSubset& thresholdsSubset,
                                                                        uint32 featureIndex) const {
    return thresholdsSubset.createRuleRefinement(*this, featureIndex);
}

std::unique_ptr<IHeadRefinement> FullIndexVector::createHeadRefinement(const IHeadRefinementFactory& factory) const {
    return factory.create(*this);
}
