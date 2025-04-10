/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"

#include <iterator>

/**
 * An iterator that provides random read-only access to a constant value.
 *
 * @tparam T The type of the constant value
 */
template<typename T>
class EqualIterator final {
    private:

        const T value_;

        uint32 index_;

    public:

        /**
         * @param value The constant value
         */
        EqualIterator(T value) : EqualIterator<T>(value, 0) {}

        /**
         * @param value         The constant value
         * @param startIndex    The index to start with
         */
        EqualIterator(T value, uint32 startIndex) : value_(value), index_(startIndex) {}

        /**
         * The type that is used to represent the difference between two iterators.
         */
        typedef int difference_type;

        /**
         * The type of the elements, the iterator provides access to.
         */
        typedef T value_type;

        /**
         * The type of a pointer to an element, the iterator provides access to.
         */
        typedef const T* pointer;

        /**
         * The type of a reference to an element, the iterator provides access to.
         */
        typedef T& reference;

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
        value_type operator[](uint32 index) const {
            return value_;
        }

        /**
         * Returns the element, the iterator currently refers to.
         *
         * @return The element, the iterator currently refers to
         */
        value_type operator*() const {
            return value_;
        }

        /**
         * Returns an iterator to the next element.
         *
         * @return A reference to an iterator that refers to the next element
         */
        EqualIterator<T>& operator++() {
            ++index_;
            return *this;
        }

        /**
         * Returns an iterator to the next element.
         *
         * @return A reference to an iterator that refers to the next element
         */
        EqualIterator<T>& operator++(int n) {
            index_++;
            return *this;
        }

        /**
         * Returns an iterator to the previous element.
         *
         * @return A reference to an iterator that refers to the previous element
         */
        EqualIterator<T>& operator--() {
            --index_;
            return *this;
        }

        /**
         * Returns an iterator to the previous element.
         *
         * @return A reference to an iterator that refers to the previous element
         */
        EqualIterator<T>& operator--(int n) {
            index_--;
            return *this;
        }

        /**
         * Returns whether this iterator and another one refer to the same element.
         *
         * @param rhs   A reference to another iterator
         * @return      True, if the iterators do not refer to the same element, false otherwise
         */
        bool operator!=(const EqualIterator<T>& rhs) const {
            return index_ != rhs.index_;
        }

        /**
         * Returns whether this iterator and another one refer to the same element.
         *
         * @param rhs   A reference to another iterator
         * @return      True, if the iterators refer to the same element, false otherwise
         */
        bool operator==(const EqualIterator<T>& rhs) const {
            return index_ == rhs.index_;
        }

        /**
         * Returns the difference between this iterator and another one.
         *
         * @param rhs   A reference to another iterator
         * @return      The difference between the iterators
         */
        difference_type operator-(const EqualIterator<T>& rhs) const {
            return (difference_type) index_ - (difference_type) rhs.index_;
        }
};
