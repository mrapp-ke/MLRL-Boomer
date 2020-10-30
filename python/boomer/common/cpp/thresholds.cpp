#include "thresholds.h"


CoverageMask::CoverageMask(uint32 numElements)
    : array_(new uint32[numElements]{0}), numElements_(numElements), target(0) {

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

bool CoverageMask::isCovered(uint32 pos) const {
    return array_[pos] == target;
}

AbstractThresholds::AbstractThresholds(std::shared_ptr<IFeatureMatrix> featureMatrixPtr,
                                       std::shared_ptr<INominalFeatureVector> nominalFeatureVectorPtr,
                                       std::shared_ptr<AbstractStatistics> statisticsPtr,
                                       std::shared_ptr<IHeadRefinementFactory> headRefinementFactoryPtr)
    : featureMatrixPtr_(featureMatrixPtr), nominalFeatureVectorPtr_(nominalFeatureVectorPtr),
      statisticsPtr_(statisticsPtr), headRefinementFactoryPtr_(headRefinementFactoryPtr) {

}

uint32 AbstractThresholds::getNumRows() const {
    return featureMatrixPtr_->getNumRows();
}

uint32 AbstractThresholds::getNumCols() const {
    return featureMatrixPtr_->getNumCols();
}

uint32 AbstractThresholds::getNumLabels() const {
    return statisticsPtr_->getNumCols();
}
