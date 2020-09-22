#pragma once

#include "arrays.h"
#include "statistics.h"
#include "tuples.h"

class BinningObserver{

    public:

        virtual void onBinUpdate(intp binIndex, IndexedFloat32* indexedValue);

};

class AbstractBinning{

    public:

        virtual ~AbstractBinning();

        virtual void createBins(uint32 numBins, IndexedFloat32Array* indexedArray, BinningObserver* observer);

};

class EqualFrequencyBinning : public AbstractBinning{

    public:

        void createBins(uint32 numBins, IndexedFloat32Array* indexedArray, BinningObserver* observer) override;

};

class EqualWidthBinning : public AbstractBinning{

    public:

        void createBins(uint32 numBins, IndexedFloat32Array* indexedArray, BinningObserver* observer) override;

};