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

std::unique_ptr<ThresholdVector> NominalFeatureBinning::createBins(FeatureInfo featureInfo,
                                                                   const FeatureVector& featureVector,
                                                                   Callback callback) const {
    uint32 numBins = featureInfo.numBins;
    std::unique_ptr<ThresholdVector> thresholdVectorPtr = std::make_unique<ThresholdVector>(numBins);

    if (numBins > 0) {
        uint32 numElements = featureVector.getNumElements();
        FeatureVector::const_iterator featureIterator = featureVector.cbegin();
        ThresholdVector::iterator thresholdIterator = thresholdVectorPtr->begin();
        float32 previousValue = featureIterator[0].value;
        thresholdIterator[0] = previousValue;
        callback(0, featureIterator[0].index, previousValue);
        uint32 binIndex = 0;

        for (uint32 i = 1; i < numElements; i++) {
            float32 currentValue = featureIterator[i].value;

            if (currentValue != previousValue) {
                previousValue = currentValue;
                binIndex++;
                thresholdIterator[binIndex] = currentValue;
            }

            callback(binIndex, featureIterator[i].index, currentValue);
        }
    }

    return thresholdVectorPtr;
}
