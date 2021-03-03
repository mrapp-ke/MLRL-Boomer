#include "common/binning/feature_binning_nominal.hpp"


IFeatureBinning::FeatureInfo NominalFeatureBinning::getFeatureInfo(FeatureVector& featureVector) const {
    FeatureInfo featureInfo;
    uint32 numElements = featureVector.getNumElements();

    if (numElements > 0) {
        featureVector.sortByValues();
        FeatureVector::const_iterator iterator = featureVector.cbegin();
        float32 previousValue = iterator[0].value;
        uint32 numDistinctValues = 1;

        for (uint32 i = 1; i < numElements; i++) {
            float32 currentValue = iterator[i].value;

            if (currentValue != previousValue) {
                numDistinctValues++;
                previousValue = currentValue;
            }
        }

        featureInfo.numBins = numDistinctValues > 1 ? numDistinctValues : 0;
    } else {
        featureInfo.numBins = 0;
    }

    return featureInfo;
}

IFeatureBinning::Result NominalFeatureBinning::createBins(FeatureInfo featureInfo,
                                                          const FeatureVector& featureVector) const {
    Result result;
    uint32 numBins = featureInfo.numBins;
    result.thresholdVectorPtr = std::make_unique<ThresholdVector>(numBins);
    uint32 numElements = featureVector.getNumElements();
    result.binIndicesPtr = std::make_unique<BinIndexVector>(numElements);

    if (numBins > 0) {
        FeatureVector::const_iterator featureIterator = featureVector.cbegin();
        ThresholdVector::iterator thresholdIterator = result.thresholdVectorPtr->begin();
        BinIndexVector::iterator binIndexIterator = result.binIndicesPtr->begin();
        float32 previousValue = featureIterator[0].value;
        thresholdIterator[0] = previousValue;
        binIndexIterator[featureIterator[0].index] = 0;
        uint32 binIndex = 0;

        for (uint32 i = 1; i < numElements; i++) {
            float32 currentValue = featureIterator[i].value;

            if (currentValue != previousValue) {
                binIndex++;
                thresholdIterator[binIndex] = currentValue;
                previousValue = currentValue;
            }

            binIndexIterator[featureIterator[i].index] = binIndex;
        }
    }

    return result;
}
