/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view.hpp"

#include <iterator>

/**
 * An iterator that provides access to the values of elements that are grouped into bins.
 *
 * @tparam ValueType    The type of the values
 * @tparam IndexType    The type of the indices that assign elements to bins
 */
template<typename ValueType, typename IndexType = const uint32>
class BinnedIterator final {
    private:

        View<IndexType> binIndexView_;

        View<ValueType> valueView_;

        uint32 index_;

    public:

        /**
         * @param binIndexView  A `View` that provides access to the indices that assign elements to bins
         * @param valueView     A `View` that provides access to the values
         * @param index         The index to start at
         */
        BinnedIterator(View<IndexType> binIndexView, View<ValueType> valueView, uint32 index)
            : binIndexView_(binIndexView), valueView_(valueView), index_(index) {}

        /**
         * The type that is used to represent the difference between two iterators.
         */
        typedef int difference_type;

        /**
         * The type of the elements, the iterator provides access to.
         */
        typedef ValueType value_type;

        /**
         * The type of a pointer to an element, the iterator provides access to.
         */
        typedef ValueType* pointer;

        /**
         * The type of a reference to an element, the iterator provides access to.
         */
        typedef ValueType& reference;

        /**
         * The tag that specifies the capabilities of the iterator.
         */
        typedef std::random_access_iterator_tag iterator_category;

        /**
         * Returns the element at a specific index.
         *
         * @param index The index of the element to be returned
         * @return      The element at the given index
         */
        reference operator[](uint32 index) const {
            uint32 binIndex = binIndexView_[index];
            return valueView_[binIndex];
        }

        /**
         * Returns the element at a specific index.
         *
         * @param index The index of the element to be returned
         * @return      The element at the given index
         */
        reference operator[](uint32 index) {
            uint32 binIndex = binIndexView_[index];
            return valueView_[binIndex];
        }

        /**
         * Returns the element, the iterator currently refers to.
         *
         * @return The element, the iterator currently refers to
         */
        reference operator*() const {
            uint32 binIndex = *binIndexView_[index_];
            return valueView_[binIndex];
        }

        /**
         * Returns the element, the iterator currently refers to.
         *
         * @return The element, the iterator currently refers to
         */
        reference operator*() {
            uint32 binIndex = *binIndexView_[index_];
            return valueView_[binIndex];
        }

        /**
         * Returns an iterator to the next element.
         *
         * @return A reference to an iterator to the next element
         */
        BinnedIterator& operator++() {
            ++index_;
            return *this;
        }

        /**
         * Returns an iterator to the next element.
         *
         * @return A reference to an iterator to the next element
         */
        BinnedIterator& operator++(int n) {
            index_++;
            return *this;
        }

        /**
         * Returns an iterator to the previous element.
         *
         * @return A reference to an iterator to the previous element
         */
        BinnedIterator& operator--() {
            --index_;
            return *this;
        }

        /**
         * Returns an iterator to the previous element.
         *
         * @return A reference to an iterator to the previous element
         */
        BinnedIterator& operator--(int n) {
            index_--;
            return *this;
        }

        /**
         * Returns whether this iterator and another one refer to the same element.
         *
         * @param rhs   A reference to another iterator
         * @return      True, if the iterators do not refer to the same element, false otherwise
         */
        bool operator!=(const BinnedIterator& rhs) const {
            return index_ != rhs.index_;
        }

        /**
         * Returns whether this iterator and another one refer to the same element.
         *
         * @param rhs   A reference to another iterator
         * @return      True, if the iterators refer to the same element, false otherwise
         */
        bool operator==(const BinnedIterator& rhs) const {
            return index_ == rhs.index_;
        }

        /**
         * Returns the difference between this iterator and another one.
         *
         * @param rhs   A reference to another iterator
         * @return      The difference between the iterators
         */
        difference_type operator-(const BinnedIterator& rhs) const {
            return (difference_type) (index_ - rhs.index_);
        }
};
