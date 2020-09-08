#pragma once

#include "arrays.h"
#include "statistics.h"

class AbstractBinning{

    public:

        virtual ~AbstractBinning();

        virtual void createBins(IndexedFloat32Array* indexedArray, BinningObserver* observer);

        void addObserver(BinningObserver* obs);

};