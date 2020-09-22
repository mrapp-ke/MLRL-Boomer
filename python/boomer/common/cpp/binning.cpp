#include "binning.h"
#include <math.h>
#include <stdexcept>

AbstractBinning::~AbstractBinning(){

}

void AbstractBinning::createBins(uint32 numBins, IndexedFloat32Array* indexedArray, BinningObserver* observer){

}


void BinningObserver::onBinUpdate(intp binIndex, IndexedFloat32* indexedValue){

};


void EqualFrequencyBinning::createBins(uint32 numBins, IndexedFloat32Array* indexedArray, BinningObserver* observer){
    //TODO: Sortieren
    //Defining length of the list, because we'll use it at least four times
    intp length = indexedArray->numElements;
    //Throwing an exception if the caller doesn't fulfil the requirement
    if(numBins > length){
        throw std::invalid_argument("numBins has to be less or equal to the length of the example array");
    }
    intp n; //number of elements per Bin
    if((length % numBins) == 0){    //if the division has no residual
        n = length/numBins;         //n is the normal division
    } else {
        n = length/numBins + 1;     //n is rounded up
    }
    //looping over bins
    intp index = 0;             //Has to be initialized for the first iteration
    float last_value = 0.0;     //Has to be initialized for the first iteration
    for(intp i = 0; i < length; i++){
        //if the value is equal to the last one it will be put in the same bin...
        if(last_value != indexedArray->data[i].value){
            index = i / n;  //... else we calculate it's own bin index
        }
        //set last value to the current one for the next iteration
        last_value = indexedArray->data[i].value;
        //notify observer
        observer->onBinUpdate(index, &indexedArray->data[i]);
    }
}


void EqualWidthBinning::createBins(uint32 numBins, IndexedFloat32Array* indexedArray, BinningObserver* observer){
    //Defining length of the list, because we'll use it at least four times
    intp length = indexedArray->numElements;
    //Throwing an exception if the caller doesn't fulfil the requirement
    if(numBins > length){
        throw std::invalid_argument("numBins has to be less or equal to the length of the example array");
    }
    //defining minimal and maximum values
    float min = indexedArray->data[0].value;
    float max = indexedArray->data[0].value;
    for(intp i = 1; i < length; i++){
        if(indexedArray->data[i].value < min){
            min = indexedArray->data[i].value;
        }else if(max < indexedArray->data[i].value){
            max = indexedArray->data[i].value;
        }
    }
    //w stands for width and determines the span of values for a bin
    intp w = intp(ceil((max - min)/numBins));
    intp index;
    for(intp i = 0; i < length; i++){
        index = floor((indexedArray->data[i].value - min) / w);
        //in some cases the calculated index can exceed the last bin, in which case we want the example in the last bin
        if(index >= numBins){
            index = numBins - 1;
        }
        //notify observer
        observer->onBinUpdate(index, &indexedArray->data[i]);
    }
}

