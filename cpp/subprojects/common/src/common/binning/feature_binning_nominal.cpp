#include "common/binning/feature_binning_nominal.hpp"
#include <unordered_map>


IFeatureBinning::Result NominalFeatureBinning::createBins(FeatureVector& featureVector) const {
    Result result;
    uint32 numElements = featureVector.getNumElements();
    result.binIndicesPtr = std::make_unique<BinIndexVector>(numElements);
    result.thresholdVectorPtr = std::make_unique<ThresholdVector>(numElements);

    if (numElements > 0) {
        FeatureVector::const_iterator featureIterator = featureVector.cbegin();
        ThresholdVector::iterator thresholdIterator = result.thresholdVectorPtr->begin();
        BinIndexVector::iterator binIndexIterator = result.binIndicesPtr->begin();
        std::unordered_map<float32, uint32> mapping;
        uint32 nextBinIndex = 0;

        for (uint32 i = 0; i < numElements; i++) {
            uint32 index = featureIterator[i].index;
            float32 currentValue = featureIterator[i].value;
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
