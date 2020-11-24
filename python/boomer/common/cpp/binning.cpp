#include "binning.h"
#include <cmath>
#include <unordered_set>


EqualFrequencyBinningImpl::EqualFrequencyBinningImpl(float32 binRatio)
    : binRatio_(binRatio) {

}

IBinning::FeatureInfo EqualFrequencyBinningImpl::getFeatureInfo(FeatureVector& featureVector) const {
    FeatureInfo featureInfo;
    uint32 numElements = featureVector.getNumElements();

    if (numElements > 0) {
        featureVector.sortByValues();
        FeatureVector::const_iterator iterator = featureVector.cbegin();
        float32 previousValue = iterator[0].value;
        uint32 numDistinctValues = 1;

        for (uint32 i = 1; i < numElements; i++) {
            float32 value = iterator[i].value;

            if (previousValue != value) {
                numDistinctValues++;
                previousValue = value;
            }
        }

        featureInfo.numBins = ceil(numDistinctValues * binRatio_);
    } else {
        featureInfo.numBins = 0;
    }

    return featureInfo;
}

void EqualFrequencyBinningImpl::createBins(FeatureInfo featureInfo, const FeatureVector& featureVector,
                                           IBinningObserver& observer) const {
    uint32 numBins = featureInfo.numBins;
    //Defining length of the list, because we'll use it at least four times
    uint32 length = featureVector.getNumElements();
    FeatureVector::const_iterator iterator = featureVector.cbegin();
    uint32 numElementsPerBin = (uint32) ceil((float) length / (float) numBins);
    //looping over bins
    uint32 binIndex = 0;  //Has to be initialized for the first iteration
    float32 previousValue = 0.0;  //Has to be initialized for the first iteration
    for (uint32 i = 0; i < length; i++) {
        float32 currentValue = iterator[i].value;
        //if the value is equal to the last one it will be put in the same bin...
        if (previousValue != currentValue) {
            binIndex = i / numElementsPerBin;  //... else we calculate it's own bin index
        }
        //set last value to the current one for the next iteration
        previousValue = currentValue;
        //notify observer
        observer.onBinUpdate(binIndex, iterator[i]);
    }
}

EqualWidthBinningImpl::EqualWidthBinningImpl(float32 binRatio)
    : binRatio_(binRatio) {

}

IBinning::FeatureInfo EqualWidthBinningImpl::getFeatureInfo(FeatureVector& featureVector) const {
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

        featureInfo.numBins = ceil(numDistinctValues * binRatio_);
        featureInfo.minValue = minValue;
        featureInfo.maxValue = maxValue;
    } else {
        featureInfo.numBins = 0;
    }

    return featureInfo;
}

void EqualWidthBinningImpl::createBins(FeatureInfo featureInfo, const FeatureVector& featureVector,
                                       IBinningObserver& observer) const {
    uint32 numBins = featureInfo.numBins;
    float32 min = featureInfo.minValue;
    float32 max = featureInfo.maxValue;
    //Defining length of the list, because we'll use it at least four times
    uint32 length = featureVector.getNumElements();
    //w stands for width and determines the span of values for a bin
    float32 spanPerBin = (max - min) / numBins;

    FeatureVector::const_iterator iterator = featureVector.cbegin();

    for (uint32 i = 0; i < length; i++) {
        float32 currentValue = iterator[i].value;
        uint32 binIndex = (uint32) floor((currentValue - min) / spanPerBin);
        //in some cases the calculated index can exceed the last bin, in which case we want the example in the last bin
        if (binIndex >= numBins) {
            binIndex = numBins - 1;
        }
        //notify observer
        observer.onBinUpdate(binIndex, iterator[i]);
    }
}
