#pragma once

#include "../../common/cpp/arrays.h"
#include "../../common/cpp/statistics.h"
#include <vector>

class AbstractBinning{

    public:

        virtual AbstractBinning();

        virtual ~AbstractBinning();

        //Ich bin mit nicht sicher, ob der observer hier gehändelt werden sollte (siehe klassische Observermethoden unten)
        virtual AbstractStatistic* createBins(indexedFloatArray* originalMatrix, intp numTargetElements, BinningObserver* observer);

        void addObserver(BinningObserver* obs);

    private:

        vector < class BinningObserver * > obs;

        void notify();

}