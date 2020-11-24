#include "feature_binning_equal_frequency.h"
#include <cmath>


EqualFrequencyBinning::EqualFrequencyBinning(float32 binRatio)
    : binRatio_(binRatio) {

}

IFeatureBinning::FeatureInfo EqualFrequencyBinning::getFeatureInfo(FeatureVector& featureVector) const {
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

void EqualFrequencyBinning::createBins(FeatureInfo featureInfo, const FeatureVector& featureVector,
                                       IBinningObserver<float32>& observer) const {
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
        observer.onBinUpdate(binIndex, iterator[i].index, currentValue);
    }
}
