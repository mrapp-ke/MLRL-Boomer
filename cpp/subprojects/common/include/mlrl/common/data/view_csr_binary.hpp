/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_matrix.hpp"

/**
 * Implements row-wise read and write access to binary values that are stored in a pre-allocated matrix in the
 * compressed sparse row (CSR) format.
 */
class BinaryCsrView : public Matrix {
    protected:

        /**
         * A pointer to an array that stores the column-indices, the non-zero elements correspond to.
         */
        uint32* colIndices_;

        /**
         * A pointer to an array that stores the indices of the first element in `colIndices_` that corresponds to a
         * certain row.
         */
        uint32* indptr_;

    public:

        /**
         * @param numRows       The number of rows in the view
         * @param numCols       The number of columns in the view
         * @param colIndices    A pointer to an array of type `uint32`, shape `(num_non_zero_values)`, that stores the
         *                      column-indices, the non-zero elements correspond to
         * @param indptr        A pointer to an array of type `uint32`, shape `(numRows + 1)`, that stores the indices
         *                      of the first element in `colIndices` that corresponds to a certain row. The index at the
         *                      last position is equal to `num_non_zero_values`
         */
        BinaryCsrView(uint32 numRows, uint32 numCols, uint32* colIndices, uint32* indptr);

        virtual ~BinaryCsrView() override {}

        /**
         * An iterator that provides read-only access to the indices in the view.
         */
        typedef typename View<uint32>::const_iterator index_const_iterator;

        /**
         * An iterator that provides access to the indices of the view and allows to modify them.
         */
        typedef typename View<uint32>::iterator index_iterator;

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

        /**
         * Returns the number of non-zero elements in the view.
         *
         * @return The number of non-zero elements
         */
        uint32 getNumNonZeroElements() const;
};
