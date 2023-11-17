#include "mlrl/common/thresholds/coverage_set.hpp"

#include "mlrl/common/rule_refinement/prediction.hpp"
#include "mlrl/common/thresholds/thresholds_subset.hpp"

CoverageSet::CoverageSet(uint32 numElements)
    : WritableVectorDecorator<AllocatedVector<uint32>>(AllocatedVector<uint32>(numElements)), numCovered_(numElements) {
    setViewToIncreasingValues(this->begin(), numElements, 0, 1);
}

CoverageSet::CoverageSet(const CoverageSet& other)
    : WritableVectorDecorator<AllocatedVector<uint32>>(AllocatedVector<uint32>(other.getNumElements())),
      numCovered_(other.numCovered_) {
    copyView(other.cbegin(), this->begin(), numCovered_);
}

uint32 CoverageSet::getNumCovered() const {
    return numCovered_;
}

void CoverageSet::setNumCovered(uint32 numCovered) {
    numCovered_ = numCovered;
}

void CoverageSet::reset() {
    numCovered_ = this->getNumElements();
    setViewToIncreasingValues(this->begin(), this->getNumElements(), 0, 1);
}

std::unique_ptr<ICoverageState> CoverageSet::copy() const {
    return std::make_unique<CoverageSet>(*this);
}

Quality CoverageSet::evaluateOutOfSample(const IThresholdsSubset& thresholdsSubset, const SinglePartition& partition,
                                         const IPrediction& head) const {
    return thresholdsSubset.evaluateOutOfSample(partition, *this, head);
}

Quality CoverageSet::evaluateOutOfSample(const IThresholdsSubset& thresholdsSubset, BiPartition& partition,
                                         const IPrediction& head) const {
    return thresholdsSubset.evaluateOutOfSample(partition, *this, head);
}

void CoverageSet::recalculatePrediction(const IThresholdsSubset& thresholdsSubset, const SinglePartition& partition,
                                        IPrediction& head) const {
    thresholdsSubset.recalculatePrediction(partition, *this, head);
}

void CoverageSet::recalculatePrediction(const IThresholdsSubset& thresholdsSubset, BiPartition& partition,
                                        IPrediction& head) const {
    thresholdsSubset.recalculatePrediction(partition, *this, head);
}
