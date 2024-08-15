/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_bit.hpp"
#include "mlrl/common/data/view_vector.hpp"
#include "mlrl/common/util/bit_functions.hpp"

#include <utility>

/**
 * A one-dimensional vector that provides random access to integer values, each with a specific number of bits, stored
 * in a pre-allocated array of a specific size.
 */
class BitVector : public BitView {
    public:

        /**
         * The number of elements in the bit vector.
         */
        uint32 numElements;

        /**
         * @param array         A pointer to an array of type `uint32` that stores the values, the bit vector should
         *                      provide access to
         * @param dimensions    The number of elements in each dimension of the bit vector
         */
        BitVector(uint32* array, std::initializer_list<uint32> dimensions)
            : BitView(array, dimensions), numElements(dimensions.begin()[0]) {}

        /**
         * @param array             A pointer to an array of type `uint32` that stores the values, the bit vector should
         *                          provide access to
         * @param numElements       The number of elements in the bit vector
         * @param numBitsPerElement The number of bits per element in the bit vector
         */
        BitVector(uint32* array, uint32 numElements, uint32 numBitsPerElement)
            : BitView(array, numElements, numBitsPerElement), numElements(numElements) {}

        /**
         * @param other A const reference to an object of type `BitVector` that should be copied
         */
        BitVector(const BitVector& other) : BitView(other), numElements(other.numElements) {}

        /**
         * @param other A reference to an object of type `BitVector` that should be moved
         */
        BitVector(BitVector&& other) : BitView(std::move(other)), numElements(other.numElements) {}

        virtual ~BitVector() override {}

        /**
         * Sets all values stored in the bit vector to zero.
         */
        void clear() {
            util::setViewToZeros(BaseView::array, util::bitArraySize<uint32>(numElements * BitView::numBitsPerElement));
        }
};

/**
 * Allocates the memory, a bit vector provides access to.
 *
 * @tparam BitVector The type of the bit vector
 */
template<typename BitVector>
class BitVectorAllocator : public BitVector {
    public:

        /**
         * @param numElements       The number of elements in the bit vector
         * @param numBitsPerElement The number of bits per element in the bit vector
         * @param init              True, if all elements in the bit vector should be value-initialized, false otherwise
         */
        explicit BitVectorAllocator(uint32 numElements, uint32 numBitsPerElement, bool init = false)
            : BitVector(util::allocateMemory<typename BitVector::value_type>(
                          util::bitArraySize<typename BitVector::value_type>(numElements * numBitsPerElement), init),
                        {numElements, numBitsPerElement}) {}

        /**
         * @param other A reference to an object of type `BitVectorAllocator` that should be copied
         */
        BitVectorAllocator(const BitVectorAllocator<BitVector>& other) : BitVector(other) {
            throw std::runtime_error("Objects of type BitVectorAllocator cannot be copied");
        }

        /**
         * @param other A reference to an object of type `BitVectorAllocator` that should be moved
         */
        BitVectorAllocator(BitVectorAllocator<BitVector>&& other) : BitVector(std::move(other)) {
            other.release();
        }

        virtual ~BitVectorAllocator() override {
            util::freeMemory(BitVector::array);
        }
};

/**
 * Allocates the memory, a `BitVector` provides access to.
 */
typedef BitVectorAllocator<BitVector> AllocatedBitVector;
