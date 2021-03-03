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

std::unique_ptr<ThresholdVector> EqualWidthFeatureBinning::createBins(FeatureInfo featureInfo,
                                                                      const FeatureVector& featureVector,
                                                                      Callback callback) const {
    uint32 numBins = featureInfo.numBins;

    if (numBins > 0) {
        float32 min = featureInfo.minValue;
        float32 max = featureInfo.maxValue;
        //Defining length of the list, because we'll use it at least four times
        uint32 length = featureVector.getNumElements();
        //w stands for width and determines the span of values for a bin
        float32 spanPerBin = (max - min) / numBins;

        FeatureVector::const_iterator iterator = featureVector.cbegin();

        for (uint32 i = 0; i < length; i++) {
            float32 currentValue = iterator[i].value;
            uint32 binIndex = (uint32) std::floor((currentValue - min) / spanPerBin);
            //in some cases the calculated index can exceed the last bin, in which case we want the example in the last bin
            if (binIndex >= numBins) {
                binIndex = numBins - 1;
            }
            //notify observer
            callback(binIndex, iterator[i].index, currentValue);
        }
    }

    return nullptr;
}
