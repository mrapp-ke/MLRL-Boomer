#pragma once

#include "arrays.h"
#include "abstract_binning.h"
#include "tuples.h"

class BinningObserver{

    public:

        virtual void onBinUpdate(intp binIndex, IndexedFloat32* indexedValue);

};