/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/sampling/random.hpp"


/**
 * Returns whether unassigned examples should be included in a sample or not, depending on whether more examples remain
 * to be sampled or more example remain to be left out of the sample.
 *
 * @param numDesiredSamples         The number of examples that remain to be sampled
 * @param numDesiredOutOfSamples    The number of examples that remain to be left out of sample
 * @param rng                       A reference to an object of type `RNG`, implementing the random number generator to
 *                                  be used
 * @return                          True, if unassigned examples should be included in the sample, false otherwise
 */
static inline bool tiebreak(uint32 numDesiredSamples, uint32 numDesiredOutOfSamples, RNG& rng) {
    if (numDesiredSamples > numDesiredOutOfSamples) {
        return true;
    } else if (numDesiredSamples < numDesiredOutOfSamples) {
        return false;
    } else {
        return rng.random(0, 2) != 0;
    }
}
