/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/array.hpp"

/**
 * A vector that stores the weights of individual bins, i.e., how many examples have been assigned to them.
 */
class BinWeightVector final : public VectorDecorator<AllocatedVector<uint32>> {
    public:

        /**
         * @param numElements The number of elements in the vector
         */
        BinWeightVector(uint32 numElements);

        /**
         * Resets all weights to zero.
         */
        void clear();

        /**
         * Increases the weight at a specific position by one.
         *
         * @param pos The position
         */
        void increaseWeight(uint32 pos);

        /**
         * Returns whether the weight at a specific position is non-zero or not.
         *
         * @param pos   The position
         * @return      True, if the weight is non-zero, false otherwise
         */
        bool operator[](uint32 pos) const;
};
