/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "types.hpp"
#include <iterator>


/**
 * An one-dimensional vector that provides random access to a fixed number of elements, corresponding to bins, stored in
 * a C-contiguous array.
 *
 * @tparam T The type of the data that is stored in the vector
 */
template<class T>
class DenseBinnedVector {

    private:

        uint32* binIndices_;

        T* array_;

        uint32 numElements_;

        uint32 numBins_;

        uint32 maxBinCapacity_;

    public:

        /**
         * Allows to iterate all elements in the vector.
         */
        class Iterator final {

            private:

                const DenseBinnedVector<T>& vector_;

                uint32 index_;

            public:

                Iterator(const DenseBinnedVector<T>& vector, uint32 index);

                typedef int difference_type;

                typedef T value_type;

                typedef T* pointer;

                typedef T reference;

                typedef std::random_access_iterator_tag iterator_category;

                reference operator[](uint32 index) const;

                reference operator*() const;

                Iterator& operator++();

                Iterator& operator++(int n);

                Iterator& operator--();

                Iterator& operator--(int n);

                bool operator!=(const Iterator& rhs) const;

                difference_type operator-(const Iterator& rhs) const;

        };

        /**
         * @param numElements   The number of elements in the vector
         * @param numBins       The number of bins
         */
        DenseBinnedVector(uint32 numElements, uint32 numBins);

        virtual ~DenseBinnedVector();

        typedef uint32* index_binned_iterator;

        typedef const uint32* index_binned_const_iterator;

        typedef T* binned_iterator;

        typedef const T* binned_const_iterator;

        typedef Iterator const_iterator;

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
         * Returns an `index_binned_iterator` to the beginning of the indices that correspond to the bins.
         *
         * @return An `index_binned_iterator` to the beginning
         */
        index_binned_iterator indices_binned_begin();

        /**
         * Returns an `index_binned_iterator` to the end of the indices that correspond to the bins.
         *
         * @return An `index_binned_iterator` to the end
         */
        index_binned_iterator indices_binned_end();

        /**
         * Returns an `index_binned_const_iterator` to the beginning of the indices that correspond to the bins.
         *
         * @return An `index_binned_const_iterator` to the beginning
         */
        index_binned_const_iterator indices_binned_cbegin() const;

        /**
         * Returns an `index_binned_const_iterator` to the end of the indices that correspond to the bins.
         *
         * @return An `index_binned_const_iterator` to the end
         */
        index_binned_const_iterator indices_binned_cend() const;

        /**
         * Returns a `binned_iterator` to the beginning of the elements that correspond to the bins.
         *
         * @return A `binned_iterator` to the beginning
         */
        binned_iterator binned_begin();

        /**
         * Returns a `binned_iterator` to the end of the elements that correspond to the bins.
         *
         * @return A `binned_iterator` to the end
         */
        binned_iterator binned_end();

        /**
         * Returns a `binned_const_iterator` to the beginning of the elements that correspond to the bins.
         *
         * @return A `binned_const_iterator` to the beginning
         */
        binned_const_iterator binned_cbegin() const;

        /**
         * Returns a `binned_const_iterator` to the end of the elements that correspond to the bins.
         *
         * @return A `binned_const_iterator` to the end
         */
        binned_const_iterator binned_cend() const;

        /**
         * Returns the number of elements in the vector.
         *
         * @return The number of elements in the vector
         */
        uint32 getNumElements() const;

        /**
         * Returns the number of bins.
         *
         * @return The number of bins
         */
        uint32 getNumBins() const;

        /**
         * Sets the number of bins.
         *
         * @param numBins       The number of bins to be set
         * @param freeMemory    True, if unused memory should be freed, if possible, false otherwise
         */
        void setNumBins(uint32 numBins, bool freeMemory);

        /**
         * Returns the value of the element at a specific position.
         *
         * @param pos   The position of the element. Must be in [0, getNumElements())
         * @return      The value of the given element
         */
        T getValue(uint32 pos) const;

};
