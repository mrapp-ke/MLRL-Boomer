/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_vector_bit.hpp"

/**
 * A vector that provides random read and write access to binary values stored in a newly allocated array in a
 * space-efficient way (see https://en.wikipedia.org/wiki/Bit_array).
 *
 * @tparam T The type of the values stored in the vector
 */
class BitVector final : public ClearableViewDecorator<BitVectorDecorator<AllocatedBitVector>> {
    public:

        /**
         * @param numBits   The number of bits in the vector
         * @param init      True, if all elements in the vector should be value-initialized, false otherwise
         */
        BitVector(uint32 numBits, bool init = false)
            : ClearableViewDecorator<BitVectorDecorator<AllocatedBitVector>>(AllocatedBitVector(numBits, init)) {}

        /**
         * @param other A reference to an object of type `AllocatedBitVector` that should be moved
         */
        BitVector(AllocatedBitVector&& other)
            : ClearableViewDecorator<BitVectorDecorator<AllocatedBitVector>>(AllocatedBitVector(std::move(other))) {}
};
