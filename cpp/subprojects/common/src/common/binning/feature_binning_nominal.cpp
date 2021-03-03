#include "common/binning/feature_binning_nominal.hpp"
#include "common/data/arrays.hpp"
#include <unordered_map>


IFeatureBinning::Result NominalFeatureBinning::createBins(FeatureVector& featureVector, uint32 numExamples) const {
    Result result;
    uint32 numElements = featureVector.getNumElements();
    result.binIndicesPtr = std::make_unique<BinIndexVector>(numExamples);
    result.thresholdVectorPtr = std::make_unique<ThresholdVector>(numElements);

    if (numElements > 0) {
        FeatureVector::const_iterator featureIterator = featureVector.cbegin();
        ThresholdVector::iterator thresholdIterator = result.thresholdVectorPtr->begin();
        BinIndexVector::iterator binIndexIterator = result.binIndicesPtr->begin();
        std::unordered_map<float32, uint32> mapping;
        uint32 nextBinIndex = 0;

        if (numElements < numExamples) {
            setArrayToValue(binIndexIterator, numExamples, BIN_INDEX_SPARSE);
        }

        for (uint32 i = 0; i < numElements; i++) {
            float32 currentValue = featureIterator[i].value;

            if (currentValue != 0) {
                uint32 index = featureIterator[i].index;
                auto mapIterator = mapping.emplace(currentValue, nextBinIndex);

                if (mapIterator.second) {
                    thresholdIterator[nextBinIndex] = currentValue;
                    binIndexIterator[index] = nextBinIndex;
                    nextBinIndex++;
                } else {
                    binIndexIterator[index] = mapIterator.first->second;
                }
            }
        }

        result.thresholdVectorPtr->setNumElements(nextBinIndex, true);
    }

    return result;
}
