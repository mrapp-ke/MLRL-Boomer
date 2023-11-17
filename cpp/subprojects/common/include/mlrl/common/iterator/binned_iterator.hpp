/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view.hpp"

#include <iterator>

/**
 * An iterator that provides read-only access to the values of elements that are grouped into bins.
 *
 * @tparam T The type of the values
 */
template<typename T>
class BinnedConstIterator final {
    private:

        View<uint32>::const_iterator binIndexIterator_;

        typename View<T>::const_iterator valueIterator_;

    public:

        /**
         * @param binIndexIterator  An iterator to the bin indices of individual elements
         * @param valueIterator     An iterator to the values of individual bins
         */
        BinnedConstIterator(View<uint32>::const_iterator binIndexIterator,
                            typename View<T>::const_iterator valueIterator)
            : binIndexIterator_(binIndexIterator), valueIterator_(valueIterator) {}

        /**
         * The type that is used to represent the difference between two iterators.
         */
        typedef int difference_type;

        /**
         * The type of the elements, the iterator provides access to.
         */
        typedef const T value_type;

        /**
         * The type of a pointer to an element, the iterator provides access to.
         */
        typedef const T* pointer;

        /**
         * The type of a reference to an element, the iterator provides access to.
         */
        typedef const T& reference;

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
            uint32 binIndex = binIndexIterator_[index];
            return valueIterator_[binIndex];
        }

        /**
         * Returns the element, the iterator currently refers to.
         *
         * @return The element, the iterator currently refers to
         */
        reference operator*() const {
            uint32 binIndex = *binIndexIterator_;
            return valueIterator_[binIndex];
        }

        /**
         * Returns an iterator to the next element.
         *
         * @return A reference to an iterator to the next element
         */
        BinnedConstIterator& operator++() {
            ++binIndexIterator_;
            return *this;
        }

        /**
         * Returns an iterator to the next element.
         *
         * @return A reference to an iterator to the next element
         */
        BinnedConstIterator& operator++(int n) {
            binIndexIterator_++;
            return *this;
        }

        /**
         * Returns an iterator to the previous element.
         *
         * @return A reference to an iterator to the previous element
         */
        BinnedConstIterator& operator--() {
            --binIndexIterator_;
            return *this;
        }

        /**
         * Returns an iterator to the previous element.
         *
         * @return A reference to an iterator to the previous element
         */
        BinnedConstIterator& operator--(int n) {
            binIndexIterator_--;
            return *this;
        }

        /**
         * Returns whether this iterator and another one refer to the same element.
         *
         * @param rhs   A reference to another iterator
         * @return      True, if the iterators do not refer to the same element, false otherwise
         */
        bool operator!=(const BinnedConstIterator& rhs) const {
            return binIndexIterator_ != rhs.binIndexIterator_;
        }

        /**
         * Returns whether this iterator and another one refer to the same element.
         *
         * @param rhs   A reference to another iterator
         * @return      True, if the iterators refer to the same element, false otherwise
         */
        bool operator==(const BinnedConstIterator& rhs) const {
            return binIndexIterator_ == rhs.binIndexIterator_;
        }

        /**
         * Returns the difference between this iterator and another one.
         *
         * @param rhs   A reference to another iterator
         * @return      The difference between the iterators
         */
        difference_type operator-(const BinnedConstIterator& rhs) const {
            return (difference_type) (binIndexIterator_ - rhs.binIndexIterator_);
        }
};
