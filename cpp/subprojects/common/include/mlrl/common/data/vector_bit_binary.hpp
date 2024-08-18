/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_vector_bit.hpp"

/**
 * An one-dimensional vector that stores binary data in a space-efficient way.
 */
class BinaryBitVector final
    : public ClearableViewDecorator<BitVectorDecorator<VectorDecorator<AllocatedBitVector<bool>>>> {
    public:

        /**
         * @param numElements   The number of elements in the vector
         * @param init          True, if all elements in the vector should be value-initialized, false otherwise
         */
        BinaryBitVector(uint32 numElements, bool init = false)
            : ClearableViewDecorator<BitVectorDecorator<VectorDecorator<AllocatedBitVector<bool>>>>(
                AllocatedBitVector<bool>(numElements, 1, init)) {}
};
