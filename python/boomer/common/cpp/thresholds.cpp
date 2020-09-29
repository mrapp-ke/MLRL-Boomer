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
                                                                                 uint32 numConditions,
                                                                                 uint32 totalSumOfWeights) {
    IndexedFloat32ArrayWrapper* indexedArrayWrapper = cacheFiltered_[featureIndex];

    if (indexedArrayWrapper == NULL) {
        indexedArrayWrapper = (IndexedFloat32ArrayWrapper*) malloc(sizeof(IndexedFloat32ArrayWrapper));
        indexedArrayWrapper->array = NULL;
        indexedArrayWrapper->numConditions = 0;
        cacheFiltered_[featureIndex] = indexedArrayWrapper;
    }

    IndexedFloat32Array* indexedArray = indexedArrayWrapper->array;

    if (indexedArray == NULL) {
        indexedArray = thresholds_->cache_[featureIndex];

        if (indexedArray == NULL) {
            indexedArray = (IndexedFloat32Array*) malloc(sizeof(IndexedFloat32Array));
            indexedArray->data = NULL;
            indexedArray->numElements = 0;
            thresholds_->cache_[featureIndex] = indexedArray;
        }
    }

    bool nominal = thresholds_->nominalFeatureVectorPtr_.get()->getValue(featureIndex);
    return new ExactRuleRefinementImpl(thresholds_->statisticsPtr_.get(), indexedArray, weights_, totalSumOfWeights,
                                       featureIndex, nominal);
}

ExactThresholdsImpl::ThresholdsSubsetImpl::RuleRefinementCallback::RuleRefinementCallback(
        ThresholdsSubsetImpl* thresholdsSubset, const uint32* coveredStatisticsMask, uint32 coveredStatisticsTarget,
        uint32 numConditions, uint32 featureIndex) {
    thresholdsSubset_ = thresholdsSubset;
    coveredStatisticsMask_ = coveredStatisticsMask;
    coveredStatisticsTarget_ = coveredStatisticsTarget;
    numConditions_ = numConditions;
    featureIndex_ = featureIndex;
}

IndexedFloat32Array* ExactThresholdsImpl::ThresholdsSubsetImpl::RuleRefinementCallback::getSortedFeatureValues() {
    // Obtain array that contains the indices of the training examples sorted according to the current feature...
    IndexedFloat32ArrayWrapper* indexedArrayWrapper = thresholdsSubset_->cacheFiltered_[featureIndex_];
    IndexedFloat32Array* indexedArray = indexedArrayWrapper->array;
    IndexedFloat32* indexedValues;

    if (indexedArray == NULL) {
        indexedArray = thresholdsSubset_->thresholds_->cache_[featureIndex_];
        indexedValues = indexedArray->data;

        if (indexedValues == NULL) {
            thresholdsSubset_->thresholds_->featureMatrixPtr_.get()->fetchFeatureValues(featureIndex_, indexedArray);
            indexedValues = indexedArray->data;
            qsort(indexedValues, indexedArray->numElements, sizeof(IndexedFloat32), &tuples::compareIndexedFloat32);
        }
    }

    // Filter indices, if only a subset of the contained examples is covered...
    if (numConditions_ > indexedArrayWrapper->numConditions) {
       // TODO
    }

    return indexedArray;
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
