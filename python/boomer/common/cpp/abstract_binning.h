#pragma once

#include "arrays.h"
#include "statistics.h"
#include "tuples.h"
#include "binning_observer.h"

class AbstractBinning{

    public:

        virtual ~AbstractBinning();

        virtual void createBins(IndexedFloat32Array* indexedArray, BinningObserver* observer);

        void addObserver(BinningObserver* obs);

};