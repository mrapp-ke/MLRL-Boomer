/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_matrix_sparse.hpp"

/**
 * A two-dimensional view that provides column-wise access to binary values stored in a matrix in the compressed sparse
 * column (CSC) format.
 */
class BinaryCscView : public BinarySparseMatrix {
    public:

        /**
         * @param indices   A pointer to an array of type `uint32`, shape `(numNonZeroValues)`, that stores the row
         *                  indices, the values in the matrix correspond to
         * @param indptr    A pointer to an array of type `uint32`, shape `(numCols + 1)`, that stores the indices of
         *                  the first element in `indices` that corresponds to a certain column. The index at the last
         *                  position must be equal to `numNonZeroValues`
         * @param numRows   The number of rows in the view
         * @param numCols   The number of columns in the view
         */
        BinaryCscView(uint32* indices, uint32* indptr, uint32 numRows, uint32 numCols)
            : BinarySparseMatrix(indices, indptr, numRows, numCols) {}

        /**
         * @param other A const reference to an object of type `BinaryCscView` that should be copied
         */
        BinaryCscView(const BinaryCscView& other) : BinarySparseMatrix(other) {}

        /**
         * @param other A reference to an object of type `BinaryCscView` that should be moved
         */
        BinaryCscView(BinaryCscView&& other) : BinarySparseMatrix(std::move(other)) {}

        virtual ~BinaryCscView() override {}

        /**
         * Returns an `index_const_iterator` to the beginning of the indices in a specific column of the matrix.
         *
         * @param column    The index of the column
         * @return          An `index_const_iterator` to the beginning of the indices
         */
        typename BinarySparseMatrix::index_const_iterator indices_cbegin(uint32 column) const {
            return &BinarySparseMatrix::indices[BinarySparseMatrix::indptr[column]];
        }

        /**
         * Returns an `index_const_iterator` to the end of the indices in a specific column of the matrix.
         *
         * @param column    The index of the column
         * @return          An `index_const_iterator` to the end of the indices
         */
        typename BinarySparseMatrix::index_const_iterator indices_cend(uint32 column) const {
            return &BinarySparseMatrix::indices[BinarySparseMatrix::indptr[column + 1]];
        }

        /**
         * Returns an `index_iterator` to the beginning of the indices in a specific column of the matrix.
         *
         * @param column    The index of the column
         * @return          An `index_iterator` to the beginning of the indices
         */
        typename BinarySparseMatrix::index_iterator indices_begin(uint32 column) {
            return &BinarySparseMatrix::indices[BinarySparseMatrix::indptr[column]];
        }

        /**
         * Returns an `index_iterator` to the end of the indices in a specific column of the matrix.
         *
         * @param column    The index of the column
         * @return          An `index_iterator` to the end of the indices
         */
        typename BinarySparseMatrix::index_iterator indices_end(uint32 column) {
            return &BinarySparseMatrix::indices[BinarySparseMatrix::indptr[column + 1]];
        }
};
