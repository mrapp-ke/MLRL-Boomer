/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_vector_bit_binary.hpp"

/**
 * An one-dimensional vector that stores binary data in a space-efficient way.
 */
class BinaryBitVector final
    : public ClearableViewDecorator<BinaryBitVectorDecorator<VectorDecorator<AllocatedBitVector>>> {
    public:

        /**
         * @param numElements   The number of elements in the vector
         * @param init          True, if all elements in the vector should be value-initialized, false otherwise
         */
        BinaryBitVector(uint32 numElements, bool init = false)
            : ClearableViewDecorator<BinaryBitVectorDecorator<VectorDecorator<AllocatedBitVector>>>(
                AllocatedBitVector(numElements, 1, init)) {}
};
