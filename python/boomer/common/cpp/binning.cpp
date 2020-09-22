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
    //TODO: Elemente mit selbem Wert sollen in den gleichen Bin
    //TODO: Sortieren
    intp length = indexedArray->numElements;
    //Mandatory block skipping the process, if the condition is already satisfied
    if(numBins > length){
        throw std::invalid_argument("numBins has to be less or equal to the length of the example array");
    }
    intp n; //number of elements per Bin
    if((length % numBins) == 0){    //if the division has no residual
        n = length/numBins;         //n is the normal dievision
    } else {
        n = length/numBins + 1;     //n is rounded up
    }
    //looping over bins
    intp index = 0;             //Has to be initialized
    float last_value = 0.0;     //Has to be initialized
    for(intp i = 0; i < length; i++){
        if(last_value != indexedArray->data[i].value){
            index = i / n;
        }
        last_value = indexedArray->data[i].value;
        observer->onBinUpdate(index, indexedArray->data[i]);
    }
}


void EqualWidthBinning::createBins(uint32 numBins, IndexedFloat32Array* indexedArray, BinningObserver* observer){
    intp length = indexedArray->numElements;
    //Mandatory block skipping the process, if the condition is already satisfied
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
    intp bound_min = floor(min);
    //w stands for width and determines the span of values for a bin
    intp w = intp(ceil((max - min)/numBins));
    intp index;
    for(intp i = 0; i < length; i++){
        index = floor((indexedArray->data[i] - min) / w);
        if(index >= numBins){
            index = numBins - 1;
        }
        observer->onBinUpdate(index, indexedArray->data[i]);
    }
}

