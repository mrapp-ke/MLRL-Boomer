#include "thresholds.h"


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

AbstractThresholds::AbstractThresholds(std::shared_ptr<IFeatureMatrix> featureMatrixPtr,
                                       std::shared_ptr<INominalFeatureMask> nominalFeatureMaskPtr,
                                       std::shared_ptr<AbstractStatistics> statisticsPtr,
                                       std::shared_ptr<IHeadRefinementFactory> headRefinementFactoryPtr)
    : featureMatrixPtr_(featureMatrixPtr), nominalFeatureMaskPtr_(nominalFeatureMaskPtr),
      statisticsPtr_(statisticsPtr), headRefinementFactoryPtr_(headRefinementFactoryPtr) {

}

uint32 AbstractThresholds::getNumExamples() const {
    return featureMatrixPtr_->getNumRows();
}

uint32 AbstractThresholds::getNumFeatures() const {
    return featureMatrixPtr_->getNumCols();
}

uint32 AbstractThresholds::getNumLabels() const {
    return statisticsPtr_->getNumLabels();
}
