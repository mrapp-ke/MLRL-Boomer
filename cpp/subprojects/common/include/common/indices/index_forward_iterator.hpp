/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/types.hpp"
#include <iterator>


/**
 * An iterator adaptor that adapts an iterator, which provides access to a fixed number of indices in increasing order,
 * such that it acts as a forward iterator that returns a boolean value for each possible index, indicating whether the
 * respective index is present in the original iterator or not.
 *
 * @tparam T The type of the iterator to be adapted
 */
template<class T>
class IndexForwardIterator final {

    private:

        T iterator_;

        T end_;

        uint32 index_;

        uint32 iteratorIndex_;

    public:

        /**
         * @param begin An iterator to the beginning of the indices
         * @param end   An iterator to the end of the indices
         * @param index The index to start at
         */
        IndexForwardIterator(T begin, T end, uint32 index)
            : iterator_(begin), end_(end), index_(index), iteratorIndex_(iterator_ != end_ ? *iterator_ : 0) {

        }

        /**
         * The type that is used to represent the difference between two iterators.
         */
        typedef int difference_type;

        /**
         * The type of the elements, the iterator provides access to.
         */
        typedef bool value_type;

        /**
         * The type of a pointer to an element, the iterator provides access to.
         */
        typedef bool* pointer;

        /**
         * The type of a reference to an element, the iterator provides access to.
         */
        typedef bool reference;

        /**
         * The tag that specifies the capabilities of the iterator.
         */
        typedef std::forward_iterator_tag iterator_category;

        /**
         * Returns the element, the iterator currently refers to.
         *
         * @return The element, the iterator currently refers to
         */
        reference operator*() const {
            return iterator_ != end_ && iteratorIndex_ == index_;
        }

        /**
         * Returns an iterator to the next element.
         *
         * @return A reference to an iterator that refers to the next element
         */
        IndexForwardIterator& operator++() {
            ++index_;

            if (iterator_ != end_ && iteratorIndex_ < index_) {
                iterator_++;
                iteratorIndex_ = *iterator_;
            }

            return *this;
        }

        /**
         * Returns an iterator to the next element.
         *
         * @return A reference to an iterator that refers to the next element
         */
        IndexForwardIterator& operator++(int n) {
            index_++;

            if (iterator_ != end_ && iteratorIndex_ < index_) {
                iterator_++;
                iteratorIndex_ = *iterator_;
            }

            return *this;
        }

        /**
         * Returns whether this iterator and another one refer to the same element.
         *
         * @param rhs   A reference to another iterator
         * @return      True, if the iterators refer to the same element, false otherwise
         */
        bool operator!=(const IndexForwardIterator& rhs) const {
            return index_ != rhs.index_;
        }

        /**
         * Returns the difference between this iterator and another one.
         *
         * @param rhs   A reference to another iterator
         * @return      The difference between the iterators
         */
        difference_type operator-(const IndexForwardIterator& rhs) const {
            return (difference_type) index_ - (difference_type) rhs.index_;
        }

};

/**
 * Creates and returns a new `IndexForwardIterator`.
 *
 * @param begin An iterator to the beginning of the indices
 * @param end   An iterator to the end of the indices
 * @param index The index to start at
 */
template<class T>
static inline IndexForwardIterator<T> make_index_forward_iterator(T begin, T end, uint32 index = 0) {
    return IndexForwardIterator<T>(begin, end, 0);
}
