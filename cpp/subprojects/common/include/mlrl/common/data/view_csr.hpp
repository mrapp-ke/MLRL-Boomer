/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"

/**
 * Implements row-wise read-only access to the values that are stored in a pre-allocated matrix in the compressed sparse
 * row (CSR) format.
 *
 * @tparam T The type of the values
 */
template<typename T>
class CsrConstView {
    protected:

        /**
         * The number of rows in the view.
         */
        const uint32 numRows_;

        /**
         * The number of columns in the view.
         */
        const uint32 numCols_;

        /**
         * A pointer to an array that stores all non-zero values.
         */
        T* data_;

        /**
         * A pointer to an array that stores the column-indices, the values in `data_` correspond to.
         */
        uint32* colIndices_;

        /**
         * A pointer to an array that stores the indices of the first element in `data_` and `colIndices_` that
         * corresponds to a certain row.
         */
        uint32* indptr_;

    public:

        /**
         * @param numRows       The number of rows in the view
         * @param numCols       The number of columns in the view
         * @param data          A pointer to an array of template type `T`, shape `(num_non_zero_values)`, that stores
         *                      all non-zero values
         * @param colIndices    A pointer to an array of type `uint32`, shape `(num_non_zero_values)`, that stores the
         *                      column-indices, the values in `data` correspond to
         * @param indptr        A pointer to an array of type `uint32`, shape `(numRows + 1)`, that stores the indices
         *                      of the first element in `data` and `colIndices` that corresponds to a certain row. The
         *                      index at the last position is equal to `num_non_zero_values`
         */
        CsrConstView(uint32 numRows, uint32 numCols, T* data, uint32* colIndices, uint32* indptr);

        virtual ~CsrConstView() {}

        /**
         * An iterator that provides read-only access to the values in the view.
         */
        typedef const T* value_const_iterator;

        /**
         * An iterator that provides read-only access to the indices in the view.
         */
        typedef const uint32* index_const_iterator;

        /**
         * Returns a `value_const_iterator` to the beginning of the values at a specific row.
         *
         * @param row   The row
         * @return      A `value_const_iterator` to the beginning of the values
         */
        value_const_iterator values_cbegin(uint32 row) const;

        /**
         * Returns a `value_const_iterator` to the end of the values at a specific row.
         *
         * @param row   The row
         * @return      A `value_const_iterator` to the end of the values
         */
        value_const_iterator values_cend(uint32 row) const;

        /**
         * Returns an `index_const_iterator` to the beginning of the indices at a specific row.
         *
         * @param row   The row
         * @return      An `index_const_iterator` to the beginning of the indices
         */
        index_const_iterator indices_cbegin(uint32 row) const;

        /**
         * Returns an `index_const_iterator` to the end of the indices at a specific row.
         *
         * @param row   The row
         * @return      An `index_const_iterator` to the end of the indices
         */
        index_const_iterator indices_cend(uint32 row) const;

        /**
         * Returns the number of non-zero elements in the view.
         *
         * @return The number of non-zero elements
         */
        uint32 getNumNonZeroElements() const;

        /**
         * Returns the number of rows in the view.
         *
         * @return The number of rows
         */
        uint32 getNumRows() const;

        /**
         * Returns the number of columns in the view.
         *
         * @return The number of columns
         */
        uint32 getNumCols() const;
};

/**
 * Implements row-wise read and write access to the values that are stored in a pre-allocated matrix in the compressed
 * sparse row (CSR) format.
 *
 * @tparam T The type of the values
 */
template<typename T>
class CsrView : public CsrConstView<T> {
    public:

        /**
         * @param numRows       The number of rows in the view
         * @param numCols       The number of columns in the view
         * @param data          A pointer to an array of template type `T`, shape `(num_non_zero_values)`, that stores
         *                      all non-zero values
         * @param colIndices    A pointer to an array of type `uint32`, shape `(num_non_zero_values)`, that stores the
         *                      column-indices, the values in `data` correspond to
         * @param indptr        A pointer to an array of type `uint32`, shape `(numRows + 1)`, that stores the indices
         *                      of the first element in `data` and `colIndices` that corresponds to a certain row. The
         *                      index at the last position is equal to `num_non_zero_values`
         */
        CsrView(uint32 numRows, uint32 numCols, T* data, uint32* colIndices, uint32* indptr);

        virtual ~CsrView() override {}

        /**
         * An iterator that provides access to the values in the view and allows to modify them.
         */
        typedef T* value_iterator;

        /**
         * An iterator that provides access to the indices in the view and allows to modify them.
         */
        typedef uint32* index_iterator;

        /**
         * Returns a `value_iterator` to the beginning of the values at a specific row.
         *
         * @param row   The row
         * @return      A `value_iterator` to the beginning of the values
         */
        value_iterator values_begin(uint32 row);

        /**
         * Returns a `value_iterator` to the end of the values at a specific row.
         *
         * @param row   The row
         * @return      A `value_iterator` to the end of the values
         */
        value_iterator values_end(uint32 row);

        /**
         * Returns an `index_iterator` to the beginning of the indices at a specific row.
         *
         * @param row   The row
         * @return      An `index_iterator` to the beginning of the indices
         */
        index_iterator indices_begin(uint32 row);

        /**
         * Returns an `index_iterator` to the end of the indices at a specific row.
         *
         * @param row   The row
         * @return      An `index_iterator` to the end of the indices
         */
        index_iterator indices_end(uint32 row);
};
