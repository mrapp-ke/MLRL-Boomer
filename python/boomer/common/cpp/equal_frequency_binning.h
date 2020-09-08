#pragma once

#include "arrays.h"
#include "statistics.h"
#include "abstract_binning.h"
#include <vector>

class EqualWidthBinning : AbstractBinning{

    private:

        intp numBins_;

    public:

        void createBins(intp numBins) override;

};