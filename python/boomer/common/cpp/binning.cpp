#include "binning.h"

AbstractBinning::~AbstractBinning(){

}

void AbstractBinning::createBins(IndexedFloat32Array* indexedArray, BinningObserver* observer){

}


void BinningObserver::onBinUpdate(intp binIndex, IndexedFloat32* indexedValue){

};


EqualFrequencyBinning::EqualFrequencyBinning(intp numBins){
    numBins_ = numBins;
}

void EqualFrequencyBinning::createBins(IndexedFloat32Array* indexedArray, BinningObserver* observer){
    intp length = indexedArray->numElements;
    //Mandatory block skipping the process, if the condition is already satisfied
    if(length <= numBins_){
        //TODO: inform observer that no approximation is necessary
        return;
    }
    intp n = length/numBins_; //number of elements per bin
    //Initializing result array | obsolete bc observer?
    //IndexedFloat32Array *results = (IndexedFloat32Array*)malloc(sizeof(IndexedFloat32Array));
    //results->numElements = numBins_;
    //IndexedFloat32 *resultData = (IndexedFloat32*)malloc(numBins_ * sizeof(IndexedFloat32));
    //results->data = resultData;
    IndexedFloat32 *tmp = (IndexedFloat32*)malloc(sizeof(IndexedFloat32));
    //looping over bins
    for(intp i = 0; i < numBins_; i++){
        tmp->value = 0;
        //looping over feature list between two bins
        for(intp j = i * n; j < ((i + 1) * n); j++){
            //if we would break out of bounds we have to break out of the loop
            if(j >= length){
                break;
            }
            //here we aggregate the values
            tmp->value = tmp->value + indexedArray->data[j].value;
        }
        tmp->index = i;
        observer->onBinUpdate(i, tmp);
        tmp->value = 0;
    }
}


EqualWidthBinning::EqualWidthBinning(intp numBins){
    numBins_ = numBins;
}

void EqualWidthBinning::createBins(IndexedFloat32Array* indexedArray, BinningObserver* observer){
    intp length = indexedArray->numElements;
    //Mandatory block skipping the process, if the condition is already satisfied
    if(length <= numBins_){
        //TODO: inform observer that no approximation is necessary
        return;
    }
     //defining minimal and maximum values
     float min = indexedArray->data[0].value;
     intp bound_min = floor(min);
     float max = indexedArray->data[indexedArray->numElements-1].value;
     //w stands for width and determines the span of values for a bin
     intp w = intp(ceil((max - min)/numBins_));
      //Initializing result array | obsolete bc observer?
     //IndexedFloat32Array *results = (IndexedFloat32Array*)malloc(sizeof(IndexedFloat32Array));
     //results->numElements = numBins_;
     //IndexedFloat32 *resultData = (IndexedFloat32*)malloc(numBins_ * sizeof(IndexedFloat32));
     //results->data = resultData;
     //defining the boundaries of bins
     intp boundaries[numBins_ + 1] {0};
     for(intp i = 0; i < numBins_ + 1; i++){
        boundaries[i] = bound_min + w * i;
     }
     IndexedFloat32 *tmp = (IndexedFloat32*)malloc(sizeof(IndexedFloat32));
     //looping over bins
     for(intp i = 0; i < numBins_; i++){
        tmp->value = 0;
        //looping over the list and adding every element in bin i
        for(intp j = 0; j < length; j++){
            if(boundaries[i]<= j && j < boundaries[i + 1]){
                tmp->value = tmp->value + indexedArray->data[j].value;
            }
        }
        tmp->index = i;
        observer->onBinUpdate(i, tmp);
     }
}

