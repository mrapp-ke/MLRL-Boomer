/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"

#include <memory>

/**
 * Implements a fast random number generator using 32 bit XOR shifts (for details, see
 * http://www.jstatsoft.org/v08/i14/paper).
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
        uint32 randomInt(uint32 min, uint32 max);

        /**
         * Generates and returns a random boolean.
         *
         * @return The random boolean that has been generated
         */
        bool randomBool();
};

/**
 * A factory that allows to create instances of the type `RNG`.
 */
class RNGFactory final {
    private:

        const uint32 randomState_;

    public:

        /**
         * @param randomState The seed to be used by the random number generators
         */
        RNGFactory(uint32 randomState);

        /**
         * Creates and returns a new object of type `RNG`.
         *
         * @return An unique pointer to an object of type `RNG` that has been created
         */
        std::unique_ptr<RNG> create() const;
};
