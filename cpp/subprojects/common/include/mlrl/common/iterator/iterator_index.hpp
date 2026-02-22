/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"

#include <iterator>

/**
 * An iterator that provides random read-only access to the indices in a continuous range.
 */
class IndexIterator final {
    private:

        uint32 index_;

    public:

        IndexIterator();

        /**
         * @param startIndex The index to start with
         */
        IndexIterator(uint32 startIndex);

        /**
         * The type that is used to represent the difference between two iterators.
         */
        using difference_type = int;

        /**
         * The type of the elements, the iterator provides access to.
         */
        using value_type = uint32;

        /**
         * The type of a pointer to an element, the iterator provides access to.
         */
        using pointer = const uint32*;

        /**
         * The type of a reference to an element, the iterator provides access to.
         */
        using reference = uint32&;

        /**
         * The tag that specifies the capabilities of the iterator.
         */
        using iterator_category = std::random_access_iterator_tag;

        /**
         * Returns the element at a specific index.
         *
         * @param index The index of the element to be returned
         * @return      The element at the given index
         */
        value_type operator[](uint32 index) const;

        /**
         * Returns the element, the iterator currently refers to.
         *
         * @return The element, the iterator currently refers to
         */
        value_type operator*() const;

        /**
         * Returns an iterator to the next element.
         *
         * @return A reference to an iterator that refers to the next element
         */
        IndexIterator& operator++();

        /**
         * Returns an iterator to the next element.
         *
         * @return A reference to an iterator that refers to the next element
         */
        IndexIterator& operator++(int n);

        /**
         * Returns an iterator to the previous element.
         *
         * @return A reference to an iterator that refers to the previous element
         */
        IndexIterator& operator--();

        /**
         * Returns an iterator to the previous element.
         *
         * @return A reference to an iterator that refers to the previous element
         */
        IndexIterator& operator--(int n);

        /**
         * Returns whether this iterator and another one refer to the same element.
         *
         * @param rhs   A reference to another iterator
         * @return      True, if the iterators do not refer to the same element, false otherwise
         */
        bool operator!=(const IndexIterator& rhs) const;

        /**
         * Returns whether this iterator and another one refer to the same element.
         *
         * @param rhs   A reference to another iterator
         * @return      True, if the iterators refer to the same element, false otherwise
         */
        bool operator==(const IndexIterator& rhs) const;

        /**
         * Returns the difference between this iterator and another one.
         *
         * @param rhs   A reference to another iterator
         * @return      The difference between the iterators
         */
        difference_type operator-(const IndexIterator& rhs) const;
};
