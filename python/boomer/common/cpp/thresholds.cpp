#include "thresholds.h"


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
