#include "indices.h"
#include "statistics.h"
#include "sub_sampling.h"
#include "thresholds.h"


DenseIndexVector::DenseIndexVector(uint32 numElements)
    : numElements_(numElements), array_(new uint32[numElements]) {

}

DenseIndexVector::~DenseIndexVector() {
    delete[] array_;
}

uint32 DenseIndexVector::getNumElements() const {
    return numElements_;
}

uint32 DenseIndexVector::getValue(uint32 pos) const {
    return array_[pos];
}

DenseIndexVector::index_iterator DenseIndexVector::indices_begin() {
    return array_;
}

DenseIndexVector::index_iterator DenseIndexVector::indices_end() {
    return &array_[numElements_];
}

DenseIndexVector::index_const_iterator DenseIndexVector::indices_cbegin() const {
    return array_;
}

DenseIndexVector::index_const_iterator DenseIndexVector::indices_cend() const {
    return &array_[numElements_];
}

std::unique_ptr<IThresholdsSubset> DenseIndexVector::createSubset(AbstractThresholds& thresholds,
                                                                  IWeightVector& weights) const {
    return thresholds.createSubset(weights, *this);
}

std::unique_ptr<IStatisticsSubset> DenseIndexVector::createSubset(const AbstractStatistics& statistics,
                                                                  uint32 numLabelIndices,
                                                                  const uint32* labelIndices) const {
    return statistics.createSubset(*this, numLabelIndices, labelIndices);
}

RangeIndexVector::Iterator::Iterator(uint32 index) {
    index_ = index;
}

uint32 RangeIndexVector::Iterator::operator[](uint32 index) const {
    return index;
}

uint32 RangeIndexVector::Iterator::operator*() const {
 return index_;
}

RangeIndexVector::Iterator& RangeIndexVector::Iterator::operator++(int n) {
    index_++;
    return *this;
}

bool RangeIndexVector::Iterator::operator!=(const RangeIndexVector::Iterator& rhs) const {
    return index_ != rhs.index_;
}

RangeIndexVector::RangeIndexVector(uint32 numElements) {
    numElements_ = numElements;
}

uint32 RangeIndexVector::getNumElements() const {
    return numElements_;
}

uint32 RangeIndexVector::getValue(uint32 pos) const {
    return pos;
}

RangeIndexVector::index_const_iterator RangeIndexVector::indices_cbegin() const {
    return RangeIndexVector::Iterator(0);
}

RangeIndexVector::index_const_iterator RangeIndexVector::indices_cend() const {
    return RangeIndexVector::Iterator(numElements_);
}

std::unique_ptr<IThresholdsSubset> RangeIndexVector::createSubset(AbstractThresholds& thresholds,
                                                                  IWeightVector& weights) const {
    return thresholds.createSubset(weights, *this);
}

std::unique_ptr<IStatisticsSubset> RangeIndexVector::createSubset(const AbstractStatistics& statistics,
                                                                  uint32 numLabelIndices,
                                                                  const uint32* labelIndices) const {
    return statistics.createSubset(*this, numLabelIndices, labelIndices);
}
