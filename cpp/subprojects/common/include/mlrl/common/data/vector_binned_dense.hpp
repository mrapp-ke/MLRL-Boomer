/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/vector_dense.hpp"
#include "mlrl/common/iterator/binned_iterator.hpp"

/**
 * An one-dimensional vector that provides random access to a fixed number of elements, corresponding to bins, stored in
 * a C-contiguous array.
 *
 * @tparam T The type of the data that is stored in the vector
 */
template<typename T>
class DenseBinnedVector {
    private:

        DenseVector<uint32> binIndices_;

        DenseVector<T> values_;

    public:

        /**
         * @param numElements   The number of elements in the vector
         * @param numBins       The number of bins
         */
        DenseBinnedVector(uint32 numElements, uint32 numBins);

        virtual ~DenseBinnedVector() {};

        /**
         * An iterator that provides access to the indices that correspond to individual bins and allows to modify them.
         */
        typedef typename DenseVector<uint32>::iterator index_iterator;

        /**
         * An iterator that provides read-only access to the indices that correspond to individual bins.
         */
        typedef typename DenseVector<uint32>::const_iterator index_const_iterator;

        /**
         * An iterator that provides access to the elements that correspond to individual bins and allows to modify
         * them.
         */
        typedef typename DenseVector<T>::iterator value_iterator;

        /**
         * An iterator that provides read-only access to the elements that correspond to individual bins.
         */
        typedef typename DenseVector<T>::const_iterator value_const_iterator;

        /**
         * An iterator that provides read-only access to the elements in the vector.
         */
        typedef BinnedConstIterator<T> const_iterator;

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
         * Returns an `index_iterator` to the beginning of the bin indices individual elements in the vector correspond
         * to.
         *
         * @return An `index_iterator` to the beginning
         */
        index_iterator indices_begin();

        /**
         * Returns an `index_iterator` to the end of the bin indices individual elements in the vector correspond to.
         *
         * @return An `index_iterator` to the end
         */
        index_iterator indices_end();

        /**
         * Returns an `index_const_iterator` to the beginning of the bin indices individual elements in the vector
         * correspond to.
         *
         * @return An `index_const_iterator` to the beginning
         */
        index_const_iterator indices_cbegin() const;

        /**
         * Returns an `index_const_iterator` to the end of the bin indices individual elements in the vector correspond
         * to.
         *
         * @return An `index_const_iterator` to the end
         */
        index_const_iterator indices_cend() const;

        /**
         * Returns a `value_iterator` to the beginning of the values that correspond to individual bins.
         *
         * @return A `value_iterator` to the beginning
         */
        value_iterator values_begin();

        /**
         * Returns a `value_iterator` to the end of the values that correspond to individual bins.
         *
         * @return A `value_iterator` to the end
         */
        value_iterator values_end();

        /**
         * Returns a `value_const_iterator` to the beginning of the values that correspond to individual bins.
         *
         * @return A `value_const_iterator` to the beginning
         */
        value_const_iterator values_cbegin() const;

        /**
         * Returns a `value_const_iterator` to the end of the values that correspond to individual bins.
         *
         * @return A `value_const_iterator` to the end
         */
        value_const_iterator values_cend() const;

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
         * Returns the number of elements in the vector.
         *
         * @return The number of elements in the vector
         */
        uint32 getNumElements() const;
};
