#include "binning_observer.h"

void BinningObserver::onBinUpdate(intp binIndex, IndexedFloat32* updatedMatrix){

}

BinningObserver::BinningObserver(AbstractBinning* toWatch){
    toWatch->addObserver(this);
}