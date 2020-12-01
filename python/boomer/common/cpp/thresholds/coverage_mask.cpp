#include "coverage_mask.h"


CoverageMask::CoverageMask(uint32 numElements)
    : array_(new uint32[numElements]{0}), numElements_(numElements), target(0) {

}

CoverageMask::CoverageMask(const CoverageMask& coverageMask)
    : array_(new uint32[coverageMask.numElements_]), numElements_(coverageMask.numElements_),
      target(coverageMask.target) {
    for (uint32 i = 0; i < numElements_; i++) {
        array_[i] = coverageMask.array_[i];
    }
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

    for (uint32 i = 0; i < numElements_; i++) {
        array_[i] = 0;
    }
}

bool CoverageMask::isCovered(uint32 pos) const {
    return array_[pos] == target;
}
