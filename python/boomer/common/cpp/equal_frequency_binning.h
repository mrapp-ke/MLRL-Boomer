#pragma once

#include "arrays.h"
#include "statistics.h"
#include "abstract_binning.h"

class EqualFrequencyBinning : public AbstractBinning{

    private:

        intp numBins_;

    public:

        EqualFrequencyBinning(intp numBins);

        void createBins(IndexedFloat32Array* indexedArray, BinningObserver* observer) override;

};