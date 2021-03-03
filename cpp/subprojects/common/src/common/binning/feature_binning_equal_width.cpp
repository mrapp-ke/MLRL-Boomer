#include "common/binning/feature_binning_equal_width.hpp"
#include "common/binning/binning.hpp"
#include <unordered_set>


EqualWidthFeatureBinning::EqualWidthFeatureBinning(float32 binRatio, uint32 minBins, uint32 maxBins)
    : binRatio_(binRatio), minBins_(minBins), maxBins_(maxBins) {

}

IFeatureBinning::FeatureInfo EqualWidthFeatureBinning::getFeatureInfo(FeatureVector& featureVector) const {
    FeatureInfo featureInfo;
    uint32 numElements = featureVector.getNumElements();

    if (numElements > 0) {
        FeatureVector::const_iterator iterator = featureVector.cbegin();
        float32 minValue = iterator[0].value;
        float32 maxValue = minValue;
        uint32 numDistinctValues = 1;
        std::unordered_set<float32> distinctValues;

        for (uint32 i = 1; i < numElements; i++) {
            float32 value = iterator[i].value;

            if (distinctValues.insert(value).second) {
                numDistinctValues++;

                if (value < minValue) {
                    minValue = value;
                }

                if (maxValue < value) {
                    maxValue = value;
                }
            }
        }

        featureInfo.numBins =
            numDistinctValues > 1 ? calculateNumBins(numDistinctValues, binRatio_, minBins_, maxBins_) : 0;
        featureInfo.minValue = minValue;
        featureInfo.maxValue = maxValue;
    } else {
        featureInfo.numBins = 0;
    }

    return featureInfo;
}

IFeatureBinning::Result EqualWidthFeatureBinning::createBins(FeatureInfo featureInfo,
                                                             FeatureVector& featureVector) const {
    Result result;
    uint32 numBins = featureInfo.numBins;
    result.thresholdVectorPtr = std::make_unique<ThresholdVector>(numBins);
    // TODO Set thresholds
    uint32 numElements = featureVector.getNumElements();
    result.binIndicesPtr = std::make_unique<BinIndexVector>(numElements);

    if (numBins > 0) {
        float32 min = featureInfo.minValue;
        float32 max = featureInfo.maxValue;
        float32 width = (max - min) / numBins;
        FeatureVector::const_iterator featureIterator = featureVector.cbegin();
        BinIndexVector::iterator binIndexIterator = result.binIndicesPtr->begin();

        for (uint32 i = 0; i < numElements; i++) {
            float32 currentValue = featureIterator[i].value;
            uint32 binIndex = (uint32) std::floor((currentValue - min) / width);

            if (binIndex >= numBins) {
                binIndex = numBins - 1;
            }

            binIndexIterator[featureIterator[i].index] = binIndex;
        }
    }

    return result;
}
