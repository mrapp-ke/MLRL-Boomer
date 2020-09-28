#include "thresholds.h"
#include <cstddef>
#include <stdlib.h>


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

ExactThresholdsImpl::ThresholdsSubsetImpl::ThresholdsSubsetImpl(ExactThresholdsImpl* thresholds,
                                                                IWeightVector* weights) {
    thresholds_ = thresholds;
    weights_ = weights;
}

ExactThresholdsImpl::ThresholdsSubsetImpl::~ThresholdsSubsetImpl() {
    std::unordered_map<uint32, IndexedFloat32ArrayWrapper*>::iterator iterator;

    for (iterator = cacheFiltered_.begin(); iterator != cacheFiltered_.end(); iterator++) {
        IndexedFloat32ArrayWrapper* indexedArrayWrapper = iterator->second;
        IndexedFloat32Array* indexedArray = indexedArrayWrapper->array;

        if (indexedArray != NULL) {
            free(indexedArray->data);
            free(indexedArray);
        }

        free(indexedArrayWrapper);
    }
}

IRuleRefinement* ExactThresholdsImpl::ThresholdsSubsetImpl::createRuleRefinement(uint32 featureIndex,
                                                                                 uint32 numConditions) {
    // TODO Implement
    return NULL;
}

ExactThresholdsImpl::ExactThresholdsImpl(std::shared_ptr<IFeatureMatrix> featureMatrixPtr,
                                         std::shared_ptr<INominalFeatureVector> nominalFeatureVectorPtr,
                                         std::shared_ptr<AbstractStatistics> statisticsPtr)
    : AbstractThresholds(featureMatrixPtr, nominalFeatureVectorPtr, statisticsPtr) {

}

ExactThresholdsImpl::~ExactThresholdsImpl() {
    std::unordered_map<uint32, IndexedFloat32Array*>::iterator iterator;

    for (iterator = cache_.begin(); iterator != cache_.end(); iterator++) {
        IndexedFloat32Array* indexedArray = iterator->second;
        free(indexedArray->data);
        free(indexedArray);
    }
}

IThresholdsSubset* ExactThresholdsImpl::createSubset(IWeightVector* weights) {
    return new ExactThresholdsImpl::ThresholdsSubsetImpl(this, weights);
}
