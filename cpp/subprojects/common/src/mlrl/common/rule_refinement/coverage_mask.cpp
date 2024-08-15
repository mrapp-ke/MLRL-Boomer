#include "mlrl/common/rule_refinement/coverage_mask.hpp"

#include "mlrl/common/rule_refinement/feature_subspace.hpp"
#include "mlrl/common/rule_refinement/prediction.hpp"

CoverageMask::CoverageMask(uint32 numElements)
    : DenseVectorDecorator<AllocatedVector<uint32>>(AllocatedVector<uint32>(numElements, true)), indicatorValue(0) {}

CoverageMask::CoverageMask(const CoverageMask& other)
    : DenseVectorDecorator<AllocatedVector<uint32>>(AllocatedVector<uint32>(other.getNumElements())),
      indicatorValue(other.indicatorValue) {
    util::copyView(other.cbegin(), this->begin(), this->getNumElements());
}

void CoverageMask::reset() {
    indicatorValue = 0;
    util::setViewToZeros(this->begin(), this->getNumElements());
}

bool CoverageMask::operator[](uint32 index) const {
    return this->view.array[index] == indicatorValue;
}
