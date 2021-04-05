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

    public:

        /**
         * @param begin An iterator to the beginning of the indices
         * @param end   An iterator to the end of the indices
         */
        IndexForwardIterator(T begin, T end);

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
        reference operator*() const;

        /**
         * Returns an iterator to the next element.
         *
         * @return A reference to an iterator that refers to the next element
         */
        IndexForwardIterator& operator++();

        /**
         * Returns an iterator to the next element.
         *
         * @return A reference to an iterator that refers to the next element
         */
        IndexForwardIterator& operator++(int n);

        /**
         * Returns whether this iterator and another one refer to the same element.
         *
         * @param rhs   A reference to another iterator
         * @return      True, if the iterators refer to the same element, false otherwise
         */
        bool operator!=(const IndexForwardIterator& rhs) const;

};

/**
 * Creates and returns a new `IndexForwardIterator`.
 *
 * @param begin An iterator to the beginning of the indices
 * @param end   An iterator to the end of the indices
 */
template<class T>
static inline IndexForwardIterator<T> make_forward_iterator(T begin, T end) {
    return IndexForwardIterator<T>(begin, end);
}
