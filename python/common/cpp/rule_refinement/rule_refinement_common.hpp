/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../data/types.hpp"


/**
 * Calculates a threshold as the average of two adjacent feature values `small` and `large`, where `small < large`.
 *
 * The threshold is calculated as `small + ((large - small) * 0.5`, instead of `(small + large) / 2`, in order to avoid
 * overflows.
 *
 * @param small The smaller of both feature values
 * @param large The larger of both feature values
 * @return      The threshold that has been calculated
 */
static inline float32 calculateThreshold(float32 small, float32 large) {
    return small + ((large - small) * 0.5);
}
