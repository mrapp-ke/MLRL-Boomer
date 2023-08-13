/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/vector_dense.hpp"

/**
 * An one-dimensional sparse vector that stores a fixed number of elements, consisting of an index and a value, in
 * C-contiguous arrays. Such a vector is similar to a `SparseArrayVector`, but uses separate arrays for storing the
 * indices and values.
 *
 * @tparam T The type of the data that is stored in the vector
 */
template<typename T>
class SparseArraysVector final : public IOneDimensionalView {
    private:

        DenseVector<uint32> indices_;

        DenseVector<T> values_;

    public:

        /**
         * @param numElements The number of elements in the vector
         */
        SparseArraysVector(uint32 numElements);

        /**
         * An iterator that provides access to the indices in the vector and allows to modify them.
         */
        typedef DenseVector<uint32>::iterator index_iterator;

        /**
         * An iterator that provides read-only access to the indices in the vector.
         */
        typedef DenseVector<uint32>::const_iterator index_const_iterator;

        /**
         * An iterator that provides access to the values in the vector and allows to modify them.
         */
        typedef typename DenseVector<T>::iterator value_iterator;

        /**
         * An iterator that provides read-only access to the values in the vector.
         */
        typedef typename DenseVector<T>::const_iterator value_const_iterator;

        /**
         * Returns an `index_iterator` to the beginning of the indices in the vector.
         *
         * @return An `index_iterator` to the beginning
         */
        index_iterator indices_begin();

        /**
         * Returns an `index_iterator` to the end of the indices in the vector.
         *
         * @return An `index_iterator` to the end
         */
        index_iterator indices_end();

        /**
         * Returns an `index_const_iterator` to the beginning of the indices in the vector.
         *
         * @return An `index_const_iterator` to the beginning
         */
        index_const_iterator indices_cbegin() const;

        /**
         * Returns an `index_const_iterator` to the end of the indices in the vector.
         *
         * @return An `index_const_iterator` to the end
         */
        index_const_iterator indices_cend() const;

        /**
         * Returns a `value_iterator` to the beginning of the values in the vector.
         *
         * @return A `value_iterator` to the beginning
         */
        value_iterator values_begin();

        /**
         * Returns a `value_iterator` to the end of the values in the vector.
         *
         * @return A `value_iterator` to the end
         */
        value_iterator values_end();

        /**
         * Returns a `value_const_iterator` to the beginning of the values in the vector.
         *
         * @return A `value_const_iterator` to the beginning
         */
        value_const_iterator values_cbegin() const;

        /**
         * Returns a `value_const_iterator` to the end of the values in the vector.
         *
         * @return A `value_const_iterator` to the end
         */
        value_const_iterator values_cend() const;

        uint32 getNumElements() const override;
};
