#include "mlrl/common/thresholds/coverage_mask.hpp"

#include "mlrl/common/rule_refinement/prediction.hpp"
#include "mlrl/common/thresholds/thresholds_subset.hpp"

CoverageMask::CoverageMask(uint32 numElements)
    : DenseVectorDecorator<AllocatedVector<uint32>>(AllocatedVector<uint32>(numElements, true)), indicatorValue_(0) {}

CoverageMask::CoverageMask(const CoverageMask& other)
    : DenseVectorDecorator<AllocatedVector<uint32>>(AllocatedVector<uint32>(other.getNumElements())),
      indicatorValue_(other.indicatorValue_) {
    copyView(other.cbegin(), this->begin(), this->getNumElements());
}

uint32 CoverageMask::getIndicatorValue() const {
    return indicatorValue_;
}

void CoverageMask::setIndicatorValue(uint32 indicatorValue) {
    indicatorValue_ = indicatorValue;
}

void CoverageMask::reset() {
    indicatorValue_ = 0;
    setViewToZeros(this->begin(), this->getNumElements());
}

bool CoverageMask::isCovered(uint32 pos) const {
    return this->view.array[pos] == indicatorValue_;
}
