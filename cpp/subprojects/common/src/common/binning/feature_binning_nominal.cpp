#include "common/binning/feature_binning_nominal.hpp"
#include <unordered_map>


IFeatureBinning::Result NominalFeatureBinning::createBins(FeatureVector& featureVector, uint32 numExamples) const {
    Result result;
    uint32 numElements = featureVector.getNumElements();
    result.binIndicesPtr = std::make_unique<BinIndexVector>(numElements);
    result.thresholdVectorPtr = std::make_unique<ThresholdVector>(numElements);

    if (numElements > 0) {
        FeatureVector::const_iterator featureIterator = featureVector.cbegin();
        ThresholdVector::iterator thresholdIterator = result.thresholdVectorPtr->begin();
        BinIndexVector::iterator binIndexIterator = result.binIndicesPtr->begin();
        std::unordered_map<float32, uint32> mapping;
        float32 currentValue = featureIterator[0].value;
        mapping.emplace(currentValue, 0);
        thresholdIterator[0] = currentValue;
        binIndexIterator[featureIterator[0].index] = 0;
        uint32 nextBinIndex = 1;

        for (uint32 i = 1; i < numElements; i++) {
            uint32 index = featureIterator[i].index;
            currentValue = featureIterator[i].value;
            auto mapIterator = mapping.emplace(currentValue, nextBinIndex);

            if (mapIterator.second) {
                thresholdIterator[nextBinIndex] = currentValue;
                binIndexIterator[index] = nextBinIndex;
                nextBinIndex++;
            } else {
                binIndexIterator[index] = mapIterator.first->second;
            }
        }

        result.thresholdVectorPtr->setNumElements(nextBinIndex, true);
    }

    return result;
}
