#include "binning.h"
#include <math.h>
#include <stdexcept>
#include <stdlib.h>


void EqualFrequencyBinningImpl::createBins(uint32 numBins, IndexedFloat32Array* indexedArray,
                                           IBinningObserver* observer) {
    //Defining length of the list, because we'll use it at least four times
    uint32 length = indexedArray->numElements;
    //Throwing an exception if the caller doesn't fulfil the requirement
    if (numBins > length) {
        throw std::invalid_argument("numBins has to be less or equal to the length of the example array");
    }
    //Sorting the array
    qsort(indexedArray, length, sizeof(IndexedFloat32), &tuples::compareIndexedFloat32);
    uint32 numElementsPerBin = (intp) round((float) length / (float) numBins);
    //looping over bins
    uint32 binIndex = 0;  //Has to be initialized for the first iteration
    float32 previousValue = 0.0;  //Has to be initialized for the first iteration
    for (uint32 i = 0; i < length; i++) {
        float32 currentValue = indexedArray->data[i].value;
        //if the value is equal to the last one it will be put in the same bin...
        if (previousValue != currentValue) {
            binIndex = i / numElementsPerBin;  //... else we calculate it's own bin index
        }
        //set last value to the current one for the next iteration
        previousValue = currentValue;
        //notify observer
        observer->onBinUpdate(binIndex, &indexedArray->data[i]);
    }
}

void EqualWidthBinningImpl::createBins(uint32 numBins, IndexedFloat32Array* indexedArray, IBinningObserver* observer) {
    //Defining length of the list, because we'll use it at least four times
    uint32 length = indexedArray->numElements;
    //Throwing an exception if the caller doesn't fulfil the requirement
    if (numBins > length) {
        throw std::invalid_argument("numBins has to be less or equal to the length of the example array");
    }
    //defining minimal and maximum values
    float32 min = indexedArray->data[0].value;
    float32 max = min;
    for (uint32 i = 1; i < length; i++) {
        float32 currentValue = indexedArray->data[i].value;

        if (currentValue < min) {
            min = currentValue;
        } else if (max < currentValue) {
            max = currentValue;
        }
    }
    //w stands for width and determines the span of values for a bin
    float32 spanPerBin = (max - min) / numBins;

    for (uint32 i = 0; i < length; i++) {
        float32 currentValue = indexedArray->data[i].value;
        uint32 binIndex = (uint32) floor((currentValue - min) / spanPerBin);
        //in some cases the calculated index can exceed the last bin, in which case we want the example in the last bin
        if (binIndex >= numBins) {
            binIndex = numBins - 1;
        }
        //notify observer
        observer->onBinUpdate(binIndex, &indexedArray->data[i]);
    }
}
