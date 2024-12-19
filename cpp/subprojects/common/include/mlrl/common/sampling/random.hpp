/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"

/**
 * Implements a fast random number generator using 32 bit XOR shifts (for details, see
 * https://www.jstatsoft.org/article/view/v008i14/916).
 */
class RNG final {
    private:

        uint32 randomState_;

    public:

        /**
         * @param randomState The seed to be used by the random number generator
         */
        RNG(uint32 randomState);

        /**
         * Generates and returns a random number in [min, max).
         *
         * @param min   The minimum number (inclusive)
         * @param max   The maximum number (exclusive)
         * @return      The random number that has been generated
         */
        uint32 random(uint32 min, uint32 max);
};
