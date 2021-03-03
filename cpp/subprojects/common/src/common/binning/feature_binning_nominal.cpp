#include "common/binning/feature_binning_nominal.hpp"
#include <unordered_set>


IFeatureBinning::Result NominalFeatureBinning::createBins(FeatureVector& featureVector) const {
    Result result;
    uint32 numElements = featureVector.getNumElements();
    result.binIndicesPtr = std::make_unique<BinIndexVector>(numElements);
    result.thresholdVectorPtr = std::make_unique<ThresholdVector>(numElements);

    if (numElements > 0) {
        FeatureVector::const_iterator featureIterator = featureVector.cbegin();
        ThresholdVector::iterator thresholdIterator = result.thresholdVectorPtr->begin();
        BinIndexVector::iterator binIndexIterator = result.binIndicesPtr->begin();
        std::unordered_set<float32> distinctValues;
        uint32 binIndex = 0;
        float32 currentValue = featureIterator[0].value;
        distinctValues.insert(currentValue);
        thresholdIterator[binIndex] = currentValue;
        binIndexIterator[featureIterator[0].index] = binIndex;

        for (uint32 i = 1; i < numElements; i++) {
            currentValue = featureIterator[i].value;

            if (distinctValues.insert(currentValue).second) {
                binIndex++;
                thresholdIterator[binIndex] = currentValue;
                binIndexIterator[featureIterator[i].index] = binIndex;
            }
        }

        result.thresholdVectorPtr->setNumElements(binIndex + 1, true);
    }

    return result;
}
