/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"

/**
 * Implements column-wise read and write access to binary values that are stored in a pre-allocated matrix in the
 * compressed sparse column (CSC) format.
 */
class BinaryCscView {
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
         * A pointer to an array that stores the row-indices, the non-zero elements correspond to.
         */
        uint32* rowIndices_;

        /**
         * A pointer to an array that stores the indices of the first element in `rowIndices_` that corresponds to a
         * certain column.
         */
        uint32* indptr_;

    public:

        /**
         * @param numRows       The number of rows in the view
         * @param numCols       The number of columns in the view
         * @param rowIndices    A pointer to an array of type `uint32`, shape `(num_non_zero_values)`, that stores the
         *                      row-indices, the non-zero elements correspond to
         * @param indptr        A pointer to an array of type `uint32`, shape `(numCols + 1)`, that stores the indices
         *                      of the first element in `rowIndices` that corresponds to a certain column. The index at
         *                      the last position is equal to `num_non_zero_values`
         */
        BinaryCscView(uint32 numRows, uint32 numCols, uint32* rowIndices, uint32* indptr);

        virtual ~BinaryCscView() {}

        /**
         * An iterator that provides read-only access to the indices in the view.
         */
        typedef const uint32* index_const_iterator;

        /**
         * An iterator that provides access to the indices in the view and allows to modify them.
         */
        typedef uint32* index_iterator;

        /**
         * Returns an `index_const_iterator` to the beginning of the indices at a specific column.
         *
         * @param col   The column
         * @return      An `index_const_iterator` to the beginning of the indices
         */
        index_const_iterator indices_cbegin(uint32 col) const;

        /**
         * Returns an `index_const_iterator` to the end of the indices at a specific column.
         *
         * @param col   The column
         * @return      An `index_const_iterator` to the end of the indices
         */
        index_const_iterator indices_cend(uint32 col) const;

        /**
         * Returns an `index_iterator` to the beginning of the indices at a specific column.
         *
         * @param col   The column
         * @return      An `index_iterator` to the beginning of the indices
         */
        index_iterator indices_begin(uint32 col);

        /**
         * Returns an `index_iterator` to the end of the indices at a specific column.
         *
         * @param col   The column
         * @return      An `index_iterator` to the end of the indices
         */
        index_iterator indices_end(uint32 col);

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
