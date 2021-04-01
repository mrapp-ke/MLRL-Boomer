/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/types.hpp"


/**
 * Defines an interface for one-dimensional vectors that provide access to weights.
 */
class IWeightVector {

    public:

        virtual ~IWeightVector() { };

        /**
         * Returns the number of non-zero weights.
         *
         * @return The number of non-zero weights
         */
        virtual uint32 getNumNonZeroWeights() const = 0;

        /**
         * Returns whether the vector contains any zero weights or not.
         *
         * @return True, if the vector contains any zero weights, false otherwise
         */
        virtual bool hasZeroWeights() const = 0;

        /**
         * Returns the weight of the example at a specific index.
         *
         * @param pos   The index of the example
         * @return      The weight of the example at the given index
         */
        virtual float64 getWeight(uint32 pos) const = 0;

};
