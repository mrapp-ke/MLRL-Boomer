#include "common/binning/feature_binning_equal_frequency.hpp"
#include "common/binning/binning.hpp"


EqualFrequencyFeatureBinning::EqualFrequencyFeatureBinning(float32 binRatio, uint32 minBins, uint32 maxBins)
    : binRatio_(binRatio), minBins_(minBins), maxBins_(maxBins) {

}

IFeatureBinning::FeatureInfo EqualFrequencyFeatureBinning::getFeatureInfo(FeatureVector& featureVector) const {
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

        featureInfo.numBins =
            numDistinctValues > 1 ? calculateNumBins(numDistinctValues, binRatio_, minBins_, maxBins_) : 0;
    } else {
        featureInfo.numBins = 0;
    }

    return featureInfo;
}

std::unique_ptr<ThresholdVector> EqualFrequencyFeatureBinning::createBins(FeatureInfo featureInfo,
                                                                          const FeatureVector& featureVector,
                                                                          Callback callback) const {
    uint32 numBins = featureInfo.numBins;

    if (numBins > 0) {
        //Defining length of the list, because we'll use it at least four times
        uint32 length = featureVector.getNumElements();
        FeatureVector::const_iterator iterator = featureVector.cbegin();
        uint32 numElementsPerBin = (uint32) std::ceil((float) length / (float) numBins);
        //looping over bins
        uint32 binIndex = 0;  //Has to be initialized for the first iteration
        float32 previousValue = 0.0;  //Has to be initialized for the first iteration
        for (uint32 i = 0; i < length; i++) {
            float32 currentValue = iterator[i].value;
            //if the value is equal to the last one it will be put in the same bin...
            if (previousValue != currentValue) {
                binIndex = i / numElementsPerBin;  //... else we calculate it's own bin index
                //set last value to the current one for the next iteration
                previousValue = currentValue;
            }
            //notify observer
            callback(binIndex, iterator[i].index, currentValue);
        }
    }

    return nullptr;
}
