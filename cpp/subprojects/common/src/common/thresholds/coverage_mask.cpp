#include "common/thresholds/coverage_mask.hpp"
#include "common/data/arrays.hpp"


CoverageMask::CoverageMask(uint32 numElements)
    : array_(new uint32[numElements]{0}), numElements_(numElements), target(0) {

}

CoverageMask::CoverageMask(const CoverageMask& coverageMask)
    : array_(new uint32[coverageMask.numElements_]), numElements_(coverageMask.numElements_),
      target(coverageMask.target) {
    copyArray(coverageMask.array_, array_, numElements_);
}

CoverageMask::~CoverageMask() {
    delete[] array_;
}

CoverageMask::iterator CoverageMask::begin() {
    return array_;
}

CoverageMask::iterator CoverageMask::end() {
    return &array_[numElements_];
}

void CoverageMask::reset() {
    target = 0;
    setArrayToZeros(array_, numElements_);
}

bool CoverageMask::isCovered(uint32 pos) const {
    return array_[pos] == target;
}

std::unique_ptr<ICoverageState> CoverageMask::copy() const {
    return std::make_unique<CoverageMask>(*this);
}
