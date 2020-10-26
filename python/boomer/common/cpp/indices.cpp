#include "indices.h"
#include "statistics.h"
#include "sub_sampling.h"
#include "thresholds.h"


DenseIndexVector::DenseIndexVector(uint32 numElements)
    : DenseVector<uint32>(numElements) {

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


RangeIndexVector::RangeIndexVector(uint32 numElements) {
    numElements_ = numElements;
}

uint32 RangeIndexVector::getNumElements() const {
    return numElements_;
}

uint32 RangeIndexVector::getValue(uint32 pos) const {
    return pos;
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
