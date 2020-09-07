#include "abstract_binning.h"

AbstractBinning::AbstractBinning(){

}

AbstractBinning::~AbstractBinning(){

}

AbstractStatistic* AbstractBinning::createBins(indexedFloatArray* originalMatrix, intp numTargetElements, BinningObserver* observer){

}

vector < class BinningObserver * > AbstractBinning::obs;

void AbstractBinning::addObserver(BinningObserver* obs){
    obs.push_back(obs);
}

void notify(){
    for(int i = 0; i < obs.size(); i++)
    {
        obs[i]->onBinUpdate();
    }
}