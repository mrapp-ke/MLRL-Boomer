#include "mlrl/common/thresholds/coverage_mask.hpp"

#include "mlrl/common/rule_refinement/prediction.hpp"
#include "mlrl/common/thresholds/thresholds_subset.hpp"

CoverageMask::CoverageMask(uint32 numElements)
    : WritableVectorDecorator<AllocatedVector<uint32>>(AllocatedVector<uint32>(numElements, true)), indicatorValue_(0) {
}

CoverageMask::CoverageMask(const CoverageMask& other)
    : WritableVectorDecorator<AllocatedVector<uint32>>(AllocatedVector<uint32>(other.view_.numElements)),
      indicatorValue_(other.indicatorValue_) {
    copyView(other.view_.array, this->view_.array, this->view_.numElements);
}

uint32 CoverageMask::getIndicatorValue() const {
    return indicatorValue_;
}

void CoverageMask::setIndicatorValue(uint32 indicatorValue) {
    indicatorValue_ = indicatorValue;
}

void CoverageMask::reset() {
    indicatorValue_ = 0;
    setViewToZeros(this->view_.array, this->view_.numElements);
}

bool CoverageMask::isCovered(uint32 pos) const {
    return this->view_.array[pos] == indicatorValue_;
}

std::unique_ptr<ICoverageState> CoverageMask::copy() const {
    return std::make_unique<CoverageMask>(*this);
}

Quality CoverageMask::evaluateOutOfSample(const IThresholdsSubset& thresholdsSubset, const SinglePartition& partition,
                                          const IPrediction& head) const {
    return thresholdsSubset.evaluateOutOfSample(partition, *this, head);
}

Quality CoverageMask::evaluateOutOfSample(const IThresholdsSubset& thresholdsSubset, BiPartition& partition,
                                          const IPrediction& head) const {
    return thresholdsSubset.evaluateOutOfSample(partition, *this, head);
}

void CoverageMask::recalculatePrediction(const IThresholdsSubset& thresholdsSubset, const SinglePartition& partition,
                                         IPrediction& head) const {
    thresholdsSubset.recalculatePrediction(partition, *this, head);
}

void CoverageMask::recalculatePrediction(const IThresholdsSubset& thresholdsSubset, BiPartition& partition,
                                         IPrediction& head) const {
    thresholdsSubset.recalculatePrediction(partition, *this, head);
}
