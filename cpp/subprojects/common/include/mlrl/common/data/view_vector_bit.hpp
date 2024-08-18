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
 *
 * @param T The type of the integer values, the view provides access to
 */
template<typename T>
class BitVector : public BitView {
    private:

        static inline constexpr uint32 NUM_BITS = util::bits<uint32>();

        inline constexpr uint32 getOffset(uint32 pos) const {
            return pos / (NUM_BITS / BitView::numBitsPerElement);
        }

        inline constexpr uint32 getNumShifts(uint32 pos) const {
            uint32 numIntegersPerElement = NUM_BITS / numBitsPerElement;
            return NUM_BITS - ((pos % numIntegersPerElement) * numBitsPerElement) - numBitsPerElement;
        }

        inline constexpr uint32 getBitMask(uint32 numShifts) const {
            uint32 ones = util::getNumBitCombinations(numBitsPerElement) - 1;
            return ones << numShifts;
        }

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
         * The type of the integer values, the view provides access to
         */
        typedef T type;

        /**
         * Sets all values stored in the bit vector to zero.
         */
        void clear() {
            util::setViewToZeros(BaseView::array,
                                 util::getBitArraySize<uint32>(numElements, BitView::numBitsPerElement));
        }

        /**
         * Returns the value of the element at a specific position.
         *
         * @param pos   The position of the element
         * @return      The value of the specified element
         */
        type operator[](uint32 pos) const {
            uint32 offset = this->getOffset(pos);
            uint32 numShifts = this->getNumShifts(pos);
            uint32 bitMask = this->getBitMask(numShifts);
            uint32 value = BaseView::array[offset];
            return static_cast<type>((value & bitMask) >> numShifts);
        }

        /**
         * Sets a value to the element at a specific position.
         *
         * @param pos   The position of the element
         * @param value The value to be set
         */
        void set(uint32 pos, type value) {
            uint32 offset = this->getOffset(pos);
            uint32 numShifts = this->getNumShifts(pos);
            uint32 bitMask = this->getBitMask(numShifts);
            BaseView::array[offset] &= ~bitMask;
            BaseView::array[offset] |= value << numShifts;
        }
};

/**
 * Allocates the memory, a `BitVector` provides access to.
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
                          util::getBitArraySize<typename BitVector::value_type>(numElements, numBitsPerElement), init),
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
 *
 * @tparam T The type of the integer values, the view provides access to
 */
template<typename T>
using AllocatedBitVector = BitVectorAllocator<BitVector<T>>;

/**
 * Provides random read and write access to integer values stored in a bit vector.
 *
 * @tparam BitVector The type of view, the bit vector is backed by
 */
template<typename BitVector>
class BitVectorDecorator : public BitVector {
    public:

        /**
         * @param view The view, the bit vector should be backed by
         */
        explicit BitVectorDecorator(typename BitVector::view_type&& view) : BitVector(std::move(view)) {}

        virtual ~BitVectorDecorator() override {}

        /**
         * Returns the value of the element at a specific position.
         *
         * @param pos   The position of the element
         * @return      The value of the specified element
         */
        typename BitVector::view_type::type operator[](uint32 pos) const {
            return this->view[pos];
        }

        /**
         * Sets a value to the element at a specific position.
         *
         * @param pos   The position of the element
         * @param value The value to be set
         */
        void set(uint32 pos, typename BitVector::view_type::type value) {
            this->view.set(pos, value);
        }
};
