/**
 * Provides classes that implement strategies for sub-sampling training examples, features or labels.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "arrays.h"
#include "data.h"

/**
 * Defines an interface for one-dimensional, potentially sparse, vectors that provide access to weights.
 */
class IWeightVector : virtual public ISparseRandomAccessVector<uint32> {

    public:

        virtual ~IWeightVector() { };

        /**
         * Returns the sum of the weights in the vector.
         *
         * @return The sum of the weights
         */
        virtual uint32 getSumOfWeights() = 0;

};
