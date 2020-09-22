#include "binning.h"
#include <math.h>

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
    if(length <= numBins){
        for(intp i = 0; i<length; i++){
            observer->onBinUpdate(i, &indexedArray->data[i]);
        }
        return;
    }
    //TODO: Aufrunden
    intp n = length/numBins; //number of elements per bin
    //looping over bins
    //Übers Orginalarray iterieren
    //orginal Index / n
    for(intp i = 0; i < numBins; i++){
        //we will need a new pointer in every iteration
        IndexedFloat32 *tmp = (IndexedFloat32*)malloc(sizeof(IndexedFloat32));
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
    }
}


void EqualWidthBinning::createBins(uint32 numBins, IndexedFloat32Array* indexedArray, BinningObserver* observer){
    //TODO: Min und Max Suche
    intp length = indexedArray->numElements;
    //Mandatory block skipping the process, if the condition is already satisfied
    if(length <= numBins){
        for(intp i = 0; i<length; i++){
            observer->onBinUpdate(i, &indexedArray->data[i]);
        }
        return;
    }
    //defining minimal and maximum values
    float min = indexedArray->data[0].value;
    intp bound_min = floor(min);
    float max = indexedArray->data[indexedArray->numElements-1].value;
    //w stands for width and determines the span of values for a bin
    intp w = intp(ceil((max - min)/numBins));
    //defining the boundaries of bins
    //TODO: Bounderies unnötig
    intp boundaries[numBins + 1] {0};
     for(intp i = 0; i < numBins + 1; i++){
        boundaries[i] = bound_min + w * i;
     }
     //looping over bins
     //TODO: Temporäres Array überflüssig
     //floor((Value - min) / w)
     //Sonderfall Index kann größer sein als der letze Index
     for(intp i = 0; i < numBins; i++){
        //we will need a new pointer in every iteration
        IndexedFloat32 *tmp = (IndexedFloat32*)malloc(sizeof(IndexedFloat32));
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

