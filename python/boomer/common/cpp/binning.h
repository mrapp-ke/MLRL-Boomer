#pragma once

#include "arrays.h"
#include "statistics.h"
#include "tuples.h"
#include <math.h>

class AbstractBinning{

    public:

        virtual ~AbstractBinning();

        virtual void createBins(IndexedFloat32Array* indexedArray, BinningObserver* observer);

};

class BinningObserver{

    public:

        virtual void onBinUpdate(intp binIndex, IndexedFloat32* indexedValue);

};

class EqualFrequencyBinning : public AbstractBinning{

    private:

        intp numBins_;

    public:

        EqualFrequencyBinning(intp numBins);

        void createBins(IndexedFloat32Array* indexedArray, BinningObserver* observer) override;

};

class EqualWidthBinning : public AbstractBinning{

    private:

        intp numBins_;

    public:

        EqualWidthBinning(intp numBins);

        void createBins(IndexedFloat32Array* indexedArray, BinningObserver* observer) override;

};