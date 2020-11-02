#include "binning.h"
#include <cmath>


EqualFrequencyBinningImpl::EqualFrequencyBinningImpl(float32 binRatio)
                                                    : binRatio_(binRatio) {

}

uint32 EqualFrequencyBinningImpl::getNumBins(FeatureVector& featureVector) const {

        return ceil(featureVector.getNumElements() * binRatio_);

}

void EqualFrequencyBinningImpl::createBins(uint32 numBins, FeatureVector& featureVector, IBinningObserver& observer) {
    //Defining length of the list, because we'll use it at least four times
    uint32 length = featureVector.getNumElements();
    //Sorting the array
    featureVector.sortByValues();
    FeatureVector::const_iterator iterator = featureVector.cbegin();
    uint32 numElementsPerBin = (intp) round((float) length / (float) numBins);
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

uint32 EqualWidthBinningImpl::getNumBins(FeatureVector& featureVector) const {

        return ceil(featureVector.getNumElements() * binRatio_);

}

void EqualWidthBinningImpl::createBins(uint32 numBins, FeatureVector& featureVector, IBinningObserver& observer) {
    //Defining length of the list, because we'll use it at least four times
    uint32 length = featureVector.getNumElements();
    //defining minimal and maximum values
    FeatureVector::const_iterator iterator = featureVector.cbegin();
    float32 min = iterator[0].value;
    float32 max = min;
    for (uint32 i = 1; i < length; i++) {
        float32 currentValue = iterator[i].value;

        if (currentValue < min) {
            min = currentValue;
        }

        if (max < currentValue) {
            max = currentValue;
        }
    }
    //w stands for width and determines the span of values for a bin
    float32 spanPerBin = (max - min) / numBins;

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
