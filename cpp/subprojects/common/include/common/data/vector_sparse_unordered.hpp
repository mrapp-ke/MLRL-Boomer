/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/indexed_value.hpp"
#include <vector>


/**
 * An one-dimensional sparse vector that stores a dynamic number of elements, consisting of an index and a value, in a
 * `std::vector`. This vector provides iterators for accessing these elements. They are not subject to a particular
 * order. In addition, for each index, this vector stores a pointer to the corresponding element. This allows to access
 * the elements that correspond to specific indices in constant time.
 *
 * A data structure of this kind was proposed in the paper "An Efficient Representation for Sparse Sets", Briggs,
 * Torczon 1993 (see https://dl.acm.org/doi/pdf/10.1145/176454.176484).
 *
 * @tparam T The type of the data that is stored in the vector
 */
template<typename T>
class SparseUnorderedVector final {

    private:

        std::vector<IndexedValue<T>> values_;

        IndexedValue<T>** ptrs_;

        uint32 maxElements_;

    public:

        /**
         * @param maxElements The maximum number of elements in the vector
         */
        SparseUnorderedVector(uint32 maxElements);

        ~SparseUnorderedVector();

        /**
         * An iterator that provides access to the elements in the vector and allows to modify them.
         */
        typedef typename std::vector<IndexedValue<T>>::iterator iterator;

        /**
         * An iterator that provides read-only access to the elements in the vector.
         */
        typedef typename std::vector<IndexedValue<T>>::const_iterator const_iterator;

        /**
         * Returns an `iterator` to the beginning of the vector.
         *
         * @return An `iterator` to the beginning
         */
        iterator begin();

        /**
         * Returns an `iterator` to the end of the vector.
         *
         * @return An `iterator` to the end
         */
        iterator end();

        /**
         * Returns a `const_iterator` to the beginning of the vector.
         *
         * @return A `const_iterator` to the beginning
         */
        const_iterator cbegin() const;

        /**
         * Returns a `const_iterator` to the end of the vector.
         *
         * @return A `const_iterator` to the end
         */
        const_iterator cend() const;

        /**
         * Returns the number of elements in the vector.
         *
         * @return The number of elements in the vector
         */
        uint32 getNumElements() const;

        /**
         * Returns the maximum number of elements in the vector.
         *
         * @return The maximum number of elements in the vector
         */
        uint32 getMaxElements() const;

        /**
         * Returns a reference to the element that corresponds to a specific index. If no such element is available, it
         * is inserted into the vector.
         *
         * @param index The index of the element to be returned
         * @return      A reference to the element that correspond to the given index
         */
        IndexedValue<T>& operator[](uint32 index);

        /**
         * Removes the element that corresponds to a specific index, if available.
         *
         * @param index The index of the element to be removed
         */
        void erase(uint32 index);

        /**
         * Removes all elements from the vector.
         */
        void clear();

};
