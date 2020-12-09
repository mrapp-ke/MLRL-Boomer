/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "types.h"
#include <forward_list>


/**
 * An one-dimensional vector that provides random access to a fixed number of elements stored in a C-contiguous array,
 * where each element may be associated with an arbitrary number of (unsorted) values.
 *
 * @tparam T The type of the values that are associated with individual elements
 */
template<class T>
class DenseMappingVector {

    public:

        typedef std::forward_list<T> Entry;

    private:

        Entry** array_;

        uint32 numElements_;

        uint32 maxCapacity_;

    public:

        /**
         * An iterator that provides access to the elements in the vector.
         */
        class Iterator {

            private:

                const DenseMappingVector<T>& vector_;

                uint32 index_;

            public:

                Iterator(const DenseMappingVector<T>& vector, uint32 index);

                Entry& operator[](uint32 index);

                Entry& operator*();

                Iterator& operator++(int n);

                bool operator!=(const Iterator& rhs) const;

        };

        /**
         * An iterator that provides read-only access to the elements in the vector.
         */
        class ConstIterator {

            private:

                const DenseMappingVector<T>& vector_;

                uint32 index_;

            public:

                ConstIterator(const DenseMappingVector<T>& vector, uint32 index);

                const Entry& operator[](uint32 index) const;

                const Entry& operator*() const;

                ConstIterator& operator++(int n);

                bool operator!=(const ConstIterator& rhs) const;

        };

        /**
         * @param numElements The number of elements in the vector
         */
        DenseMappingVector(uint32 numElements);

        ~DenseMappingVector();

        typedef Iterator iterator;

        typedef ConstIterator const_iterator;

        /**
         * Returns an `iterator` to the beginning of the elements in the vector.
         *
         * @return An `iterator` to the beginning
         */
        iterator begin();

        /**
         * Returns an `iterator` to the end of the elements in the vector.
         *
         * @return An `iterator` to the end
         */
        iterator end();

        /**
         * Returns a `const_iterator` to the beginning of the elements in the vector.
         *
         * @return A `const_iterator` to the beginning
         */
        const_iterator cbegin() const;

        /**
         * Returns a `const_iterator` to the end of the elements in the vector.
         *
         * @return A `const_iterator` to the end
         */
        const_iterator cend() const;

        /**
         * Returns the number of elements in the vector.
         *
         * @return The number of elements
         */
        uint32 getNumElements() const;

        /**
         * Sets the number of elements in the vector.
         *
         * @param numElements   The number of elements to be set
         * @param freeMemory    True, if unused memory should be freed, if possible, false otherwise
         */
        void setNumElements(uint32 numElements, bool freeMemory);

        /**
         * Returns the element at a specific position.
         *
         * @param pos   The position of the element to be returned
         * @return      A reference to an object of type `Entry` that corresponds to the given position
         */
        Entry& getEntry(uint32 pos);

        /**
         * Removes all values that are associated with the elements in the vector.
         */
        void clear();

};
