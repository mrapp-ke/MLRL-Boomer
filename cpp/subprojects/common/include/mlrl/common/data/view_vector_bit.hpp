/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_vector.hpp"

#include <climits>

/**
 * A one-dimensional view that provides access to binary values stored in a pre-allocated array in a space-efficient way
 * (see https://en.wikipedia.org/wiki/Bit_array).
 */
class MLRLCOMMON_API BitView : public Vector<uint32> {
    public:

        /**
         * The number of bits that can be stored per individual element in the array this view is backed by.
         */
        static inline constexpr uint32 BITS_PER_ELEMENT = static_cast<uint32>(CHAR_BIT * sizeof(uint32));

    private:

        static inline constexpr uint32 calculateNumElements(uint32 numBits) {
            return numBits / BITS_PER_ELEMENT + (numBits % BITS_PER_ELEMENT != 0);
        }

    public:

        /**
         * The number of bits in the view.
         */
        const uint32 numBits;

        /**
         * @param array     A pointer to an array of type `uint32` that stores the values, the view should provide
         *                  access to
         * @param numBits   The number of bits in the view
         */
        BitView(uint32* array, uint32 numBits)
            : Vector<uint32>(array, calculateNumElements(numBits)), numBits(numBits) {}

        /**
         * @param other A const reference to an object of type `BitView` that should be copied
         */
        BitView(const BitView& other) : Vector<uint32>(other), numBits(other.numBits) {}

        /**
         * @param other A reference to an object of type `BitView` that should be moved
         */
        BitView(BitView&& other) : Vector<uint32>(std::move(other)), numBits(other.numBits) {}

        virtual ~BitView() override {}
};

/**
 * Allocates the memory, a `BitView` provides access to.
 *
 * @tparam View The type of the view
 */
template<typename View>
class MLRLCOMMON_API BitVectorAllocator : public Allocator<View> {
    public:

        /**
         * @param numBits   The number of bits in the vector
         * @param init      True, if all elements in the view should be value-initialized, false otherwise
         */
        explicit BitVectorAllocator(uint32 numBits, bool init = false) : Allocator<View>(numBits, init) {}

        /**
         * @param other A reference to an object of type `BitVectorAllocator` that should be copied
         */
        BitVectorAllocator(const BitVectorAllocator<View>& other) : Allocator<View>(other) {}

        /**
         * @param other A reference to an object of type `BitVectorAllocator` that should be moved
         */
        BitVectorAllocator(BitVectorAllocator<View>&& other) : Allocator<View>(std::move(other)) {}

        virtual ~BitVectorAllocator() override {}
};

/**
 * Allocates the memory, a `BitView` provides access to.
 */
typedef BitVectorAllocator<BitView> AllocatedBitVector;

/**
 * A vector that stores binary values in a space-efficient way (see https://en.wikipedia.org/wiki/Bit_array).
 *
 * @tparam View The type of view, the vector is backed by
 */
template<typename View>
class MLRLCOMMON_API BitVectorDecorator : public ViewDecorator<View> {
    private:

        static inline constexpr uint32 calculateOffset(uint32 pos) {
            return pos / BitView::BITS_PER_ELEMENT;
        }

        static inline constexpr uint32 createBitMask(uint32 pos) {
            return 1U << (pos % BitView::BITS_PER_ELEMENT);
        }

    public:

        /**
         * @param view The view, the vector should be backed by
         */
        explicit BitVectorDecorator(View&& view) : ViewDecorator<View>(std::move(view)) {}

        virtual ~BitVectorDecorator() override {}

        /**
         * Returns the number of bits in the vector.
         *
         * @return The number of bits in the vector
         */
        uint32 getNumElements() const {
            return this->view.numBits;
        }

        /**
         * Returns whether the bit at a specific position is set or unset.
         *
         * @param pos   The position of the bit
         * @return      True, if the bit is set, false, if it is unset
         */
        bool operator[](uint32 pos) const {
            return this->view.array[calculateOffset(pos)] & createBitMask(pos);
        }

        /**
         * Sets or unsets the bit at a specific position.
         *
         * @param pos   The position of the bit
         * @param set   True, if the bit should be set, false, if it should be unset
         */
        void set(uint32 pos, bool set) {
            if (set) {
                this->view.array[calculateOffset(pos)] |= createBitMask(pos);
            } else {
                this->view.array[calculateOffset(pos)] &= ~createBitMask(pos);
            }
        }
};
