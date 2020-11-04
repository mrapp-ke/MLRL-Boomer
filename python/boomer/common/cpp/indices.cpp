#include "indices.h"
#include "statistics.h"
#include "sub_sampling.h"
#include "thresholds.h"


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

std::unique_ptr<IStatisticsSubset> PartialIndexVector::createSubset(const AbstractStatistics& statistics) const {
    return statistics.createSubset(*this);
}

std::unique_ptr<IRuleRefinement> PartialIndexVector::createRuleRefinement(IThresholdsSubset& thresholdsSubset,
                                                                          uint32 featureIndex) const {
    return thresholdsSubset.createRuleRefinement(*this, featureIndex);
}

std::unique_ptr<IHeadRefinement> PartialIndexVector::createHeadRefinement(const IHeadRefinementFactory& factory) const {
    return factory.create(*this);
}

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

std::unique_ptr<IStatisticsSubset> FullIndexVector::createSubset(const AbstractStatistics& statistics) const {
    return statistics.createSubset(*this);
}

std::unique_ptr<IRuleRefinement> FullIndexVector::createRuleRefinement(IThresholdsSubset& thresholdsSubset,
                                                                        uint32 featureIndex) const {
    return thresholdsSubset.createRuleRefinement(*this, featureIndex);
}

std::unique_ptr<IHeadRefinement> FullIndexVector::createHeadRefinement(const IHeadRefinementFactory& factory) const {
    return factory.create(*this);
}
