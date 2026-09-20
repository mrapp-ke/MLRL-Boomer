/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view.hpp"

#include <algorithm>
#include <climits>
#include <iterator>

/**
 * An one-dimensional view that provides access to binary values stored in a pre-allocated array in a space-efficient
 * way (see https://en.wikipedia.org/wiki/Bit_array).
 */
class MLRLCOMMON_API BitView {
    public:

        /**
         * The number of bits that can be stored per individual element in the array this view is backed by.
         */
        static inline constexpr uint32 BITS_PER_ELEMENT = static_cast<uint32>(CHAR_BIT * sizeof(uint32));

        /**
         * Calculates and returns the number of elements needed to store a specific number of bits.
         *
         * @param numBits   The number of bits
         * @return          The number of elements needed
         */
        static inline constexpr uint32 calculateNumElements(uint32 numBits) {
            return numBits / BITS_PER_ELEMENT + (numBits % BITS_PER_ELEMENT != 0);
        }

    private:

        static inline constexpr uint32 calculateOffset(uint32 pos) {
            return pos / BitView::BITS_PER_ELEMENT;
        }

        static inline constexpr uint32 createBitMask(uint32 pos) {
            return 1U << (pos % BitView::BITS_PER_ELEMENT);
        }

    public:

        /**
         * An iterator that provides random read-only access to the binary values in a `BitView`.
         */
        class ConstIterator final {
            private:

                const BitView& view_;

                uint32 index_;

            public:

                /**
                 * @param view          A reference to an object of type `BitView`, the iterator should provide access
                 *                      to
                 * @param startIndex    The index to start at
                 */
                ConstIterator(const BitView& view, uint32 startIndex = 0) : view_(view), index_(startIndex) {}

                /**
                 * The type that is used to represent the difference between two iterators.
                 */
                using difference_type = int;

                /**
                 * The type of the elements, the iterator provides access to.
                 */
                using value_type = const bool;

                /**
                 * The type of a pointer to an element, the iterator provides access to.
                 */
                using pointer = const bool*;

                /**
                 * The type of a reference to an element, the iterator provides access to.
                 */
                using reference = const bool&;

                /**
                 * The tag that specifies the capabilities of the iterator.
                 */
                using iterator_category = std::random_access_iterator_tag;

                /**
                 * Returns whether the bit at a specific index is set or unset.
                 *
                 * @param index The index of the element to be returned
                 * @return      True if the bit at the given index is set, false, if it is unset
                 */
                value_type operator[](uint32 index) const {
                    return view_[index];
                }

                /**
                 * Returns the element, the iterator currently refers to.
                 *
                 * @return The element, the iterator currently refers to
                 */
                value_type operator*() {
                    return view_[index_];
                }

                /**
                 * Returns an iterator to the next element.
                 *
                 * @return A reference to an iterator that refers to the next element
                 */
                ConstIterator& operator++() {
                    index_++;
                    return *this;
                }

                /**
                 * Returns an iterator to the next element.
                 *
                 * @return A reference to an iterator that refers to the next element
                 */
                ConstIterator& operator++(int n) {
                    ++index_;
                    return *this;
                }

                /**
                 * Returns an iterator to the previous element.
                 *
                 * @return A reference to an iterator that refers to the previous element
                 */
                ConstIterator& operator--() {
                    index_--;
                    return *this;
                }

                /**
                 * Returns an iterator to the previous element.
                 *
                 * @return A reference to an iterator that refers to the previous element
                 */
                ConstIterator& operator--(int n) {
                    --index_;
                    return *this;
                }

                /**
                 * Returns whether this iterator and another one refer to the same element.
                 *
                 * @param rhs   A reference to another iterator
                 * @return      True, if the iterators do not refer to the same element, false otherwise
                 */
                bool operator!=(const ConstIterator& rhs) const {
                    return index_ != rhs.index_;
                }

                /**
                 * Returns whether this iterator and another one refer to the same element.
                 *
                 * @param rhs   A reference to another iterator
                 * @return      True, if the iterators refer to the same element, false otherwise
                 */
                bool operator==(const ConstIterator& rhs) const {
                    return index_ == rhs.index_;
                }

                /**
                 * Returns the difference between this iterator and another one.
                 *
                 * @param rhs   A reference to another iterator
                 * @return      The difference between the iterators
                 */
                difference_type operator-(const ConstIterator& rhs) const {
                    return (difference_type) index_ - (difference_type) rhs.index_;
                }
        };

        /**
         * A pointer to the array that stores the values, the view provides access to.
         */
        uint32* array;

        /**
         * The number of bits in the view.
         */
        const uint32 numBits;

        /**
         * @param array     A pointer to an array of type `uint32` that stores the values, the view should provide
         *                  access to
         * @param numBits   The number of bits in the view
         */
        BitView(uint32* array, uint32 numBits) : array(array), numBits(numBits) {}

        /**
         * @param other A const reference to an object of type `BitView` that should be copied
         */
        BitView(const BitView& other) : array(other.array), numBits(other.numBits) {}

        /**
         * @param other A reference to an object of type `BitView` that should be moved
         */
        BitView(BitView&& other) : array(other.array), numBits(other.numBits) {}

        virtual ~BitView() {}

        /**
         * The type of the array, the view provides access to.
         */
        using value_type = uint32;

        /**
         * An iterator that provides read-only access to the binary values in the vector.
         */
        using const_iterator = ConstIterator;

        /**
         * Returns a `const_iterator` to the beginning of the binary values in the vector.
         *
         * @return A `const_iterator` to the beginning
         */
        const_iterator cbegin() const {
            return ConstIterator(*this);
        }

        /**
         * Returns a `const_iterator` to the end of the binary values in the vector.
         *
         * @return A `const_iterator` to the end
         */
        const_iterator cend() const {
            return ConstIterator(*this, numBits);
        }

        /**
         * Returns whether the bit at a specific position is set or unset.
         *
         * @param pos   The position of the bit
         * @return      True, if the bit is set, false, if it is unset
         */
        bool operator[](uint32 pos) const {
            return this->array[calculateOffset(pos)] & createBitMask(pos);
        }

        /**
         * Sets or unsets the bit at a specific position.
         *
         * @param pos   The position of the bit
         * @param set   True, if the bit should be set, false, if it should be unset
         */
        void set(uint32 pos, bool set) {
            if (set) {
                this->array[calculateOffset(pos)] |= createBitMask(pos);
            } else {
                this->array[calculateOffset(pos)] &= ~createBitMask(pos);
            }
        }

        /**
         * Sets all values stored in the view to zero.
         */
        void clear() {
            std::fill(array, &array[calculateNumElements(numBits)], (uint32) 0);
        }

        /**
         * Releases the ownership of the array that stores the values, the view provides access to. As a result, the
         * behavior of this view becomes undefined and it should not be used anymore. The caller is responsible for
         * freeing the memory that is occupied by the array.
         *
         * @return A pointer to the array that stores the values, the view provided access to
         */
        value_type* release() {
            value_type* ptr = array;
            array = nullptr;
            return ptr;
        }
};

/**
 * Allocates the memory, a `BitView` provides access to.
 *
 * @tparam View             The type of the view
 * @tparam MemoryAllocator  The type of the memory allocator to be used
 */
template<typename View, typename MemoryAllocator = DefaultMemoryAllocator>
class MLRLCOMMON_API BitVectorAllocator : public View {
    public:

        /**
         * @param numBits   The number of bits in the vector
         * @param init      True, if all elements in the view should be value-initialized, false otherwise
         */
        explicit BitVectorAllocator(uint32 numBits, bool init = false)
            : View(MemoryAllocator::template allocateMemory<uint32>(BitView::calculateNumElements(numBits), init),
                   numBits) {}

        /**
         * @param other A reference to an object of type `BitVectorAllocator` that should be copied
         */
        BitVectorAllocator(const BitVectorAllocator<View, MemoryAllocator>& other) = delete;

        /**
         * @param other A reference to an object of type `BitVectorAllocator` that should be moved
         */
        BitVectorAllocator(BitVectorAllocator<View, MemoryAllocator>&& other) : View(std::move(other)) {
            other.release();
        }

        virtual ~BitVectorAllocator() override {
            MemoryAllocator::freeMemory(View::array);
        }
};

/**
 * Allocates the memory, a `BitView` provides access to.
 *
 * @tparam MemoryAllocator The type of the memory allocator to be used
 */
template<typename MemoryAllocator = DefaultMemoryAllocator>
using AllocatedBitVector = BitVectorAllocator<BitView, MemoryAllocator>;

/**
 * A vector that stores binary values in a `BitView`.
 *
 * @tparam View The type of the `BitView`, the vector is backed by
 */
template<typename View>
class MLRLCOMMON_API BitVectorDecorator : public ViewDecorator<View> {
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
};

/**
 * Provides random read and write access to values stored in a bit vector.
 *
 * @tparam View The type of view, the vector is backed by
 */
template<typename BitVector>
class MLRLCOMMON_API IndexableBitVectorDecorator : public BitVector {
    public:

        /**
         * @param view The view, the vector should be backed by
         */
        explicit IndexableBitVectorDecorator(typename BitVector::view_type&& view) : BitVector(std::move(view)) {}

        virtual ~IndexableBitVectorDecorator() override {}

        /**
         * Returns whether the bit at a specific position is set or unset.
         *
         * @param pos   The position of the bit
         * @return      True, if the bit is set, false, if it is unset
         */
        bool operator[](uint32 pos) const {
            return this->view[pos];
        }

        /**
         * Sets or unsets the bit at a specific position.
         *
         * @param pos   The position of the bit
         * @param set   True, if the bit should be set, false, if it should be unset
         */
        void set(uint32 pos, bool set) {
            this->view.set(pos, set);
        }
};
