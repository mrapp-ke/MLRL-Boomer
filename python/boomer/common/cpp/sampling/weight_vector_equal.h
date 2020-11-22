/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "weight_vector.h"


/**
 * An one-dimensional that provides random access to a fixed number of equal weights.
 */
class EqualWeightVector : public IWeightVector {

    private:

        uint32 numElements_;

    public:

        /**
         * @param numTotalElements The number of elements in the vector. Must be at least 1
         */
        EqualWeightVector(uint32 numElements);

        bool hasZeroWeights() const override;

        uint32 getWeight(uint32 pos) const override;

        uint32 getSumOfWeights() const override;

};
