#include "thresholds.h"


AbstractThresholds::AbstractThresholds(std::shared_ptr<IFeatureMatrix> featureMatrixPtr,
                                       std::shared_ptr<INominalFeatureVector> nominalFeatureVectorPtr,
                                       std::shared_ptr<AbstractStatistics> statisticsPtr) {
    featureMatrixPtr_ = featureMatrixPtr;
    nominalFeatureVectorPtr_ = nominalFeatureVectorPtr;
    statisticsPtr_ = statisticsPtr;
}

uint32 AbstractThresholds::getNumRows() {
    return featureMatrixPtr_.get()->getNumRows();
}

uint32 AbstractThresholds::getNumCols() {
    return featureMatrixPtr_.get()->getNumCols();
}
