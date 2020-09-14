#pragma once

#include "arrays.h"
#include "statistics.h"
#include "abstract_binning.h"

class EqualWidthBinning : public AbstractBinning{

    private:

        intp numBins_;

    public:

        EqualWidthBinning(intp numBins);

        void createBins(IndexedFloat32Array* indexedArray, BinningObserver* observer) override;

};