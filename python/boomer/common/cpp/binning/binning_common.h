/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../data/types.h"
#include <cmath>


/**
 * Calculates and returns the number of bins to be used, based on the number of values available, a percentage that
 * specifies how many bins should be used, and a minimum and maximum number of bins.
 *
 * @param numValues The number of values available
 * @param binRatio  A percentage that specifies how many bins should be used
 * @param minBins   The minimum number of bins
 * @param maxBins   The maximum number of bins or a value <= `minBins`, if the maximum number should not be restricted
 */
static inline uint32 calculateNumBins(uint32 numValues, float32 binRatio, uint32 minBins, uint32 maxBins) {
    uint32 numBins = std::ceil(binRatio * numValues);

    if (numBins < minBins) {
        return minBins;
    } else if (maxBins > minBins && numBins > maxBins) {
        return maxBins;
    } else {
        return numBins;
    }
}
