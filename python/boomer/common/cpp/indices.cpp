#include "indices.h"
#include "statistics.h"
#include "sub_sampling.h"
#include "thresholds.h"
#include <cstdlib>


PartialIndexVector::PartialIndexVector(uint32 numElements)
    : numElements_(numElements), array_((uint32*) malloc(numElements * sizeof(uint32))) {

}

PartialIndexVector::~PartialIndexVector() {
    free(array_);
}

bool PartialIndexVector::isPartial() const {
    return true;
}

uint32 PartialIndexVector::getNumElements() const {
    return numElements_;
}

void PartialIndexVector::setNumElements(uint32 numElements) {
    if (numElements != numElements_) {
        numElements_ = numElements;
        array_ = (uint32*) realloc(array_, numElements * sizeof(uint32));
    }
}

uint32 PartialIndexVector::getIndex(uint32 pos) const {
    return array_[pos];
}

PartialIndexVector::index_iterator PartialIndexVector::indices_begin() {
    return array_;
}

PartialIndexVector::index_iterator PartialIndexVector::indices_end() {
    return &array_[numElements_];
}

PartialIndexVector::index_const_iterator PartialIndexVector::indices_cbegin() const {
    return array_;
}

PartialIndexVector::index_const_iterator PartialIndexVector::indices_cend() const {
    return &array_[numElements_];
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

FullIndexVector::index_const_iterator FullIndexVector::indices_cbegin() const {
    return FullIndexVector::Iterator(0);
}

FullIndexVector::index_const_iterator FullIndexVector::indices_cend() const {
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
