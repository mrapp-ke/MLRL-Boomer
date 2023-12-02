/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_matrix_sparse_binary.hpp"
#include "mlrl/common/data/view_vector.hpp"

/**
 * A two-dimensional view that provides row-wise access to binary values stored in a matrix in the compressed sparse row
 * (CSR) format.
 */
class BinaryCsrView : public BinarySparseMatrix {
    public:

        /**
         * @param indices   A pointer to an array of type `uint32`, shape `(numNonZeroValues)`, that stores the column
         *                  indices, the values in the matrix correspond to
         * @param indptr    A pointer to an array of type `uint32`, shape `(numRows + 1)`, that stores the indices of
         *                  the first element in `indices` that corresponds to a certain row. The index at the last
         *                  position must be equal to `numNonZeroValues`
         * @param numRows   The number of rows in the view
         * @param numCols   The number of columns in the view
         */
        BinaryCsrView(uint32* indices, uint32* indptr, uint32 numRows, uint32 numCols)
            : BinarySparseMatrix(indices, indptr, numRows, numCols) {}

        /**
         * @param other A const reference to an object of type `BinaryCsrView` that should be copied
         */
        BinaryCsrView(const BinaryCsrView& other) : BinarySparseMatrix(other) {}

        /**
         * @param other A reference to an object of type `BinaryCsrView` that should be moved
         */
        BinaryCsrView(BinaryCsrView&& other) : BinarySparseMatrix(std::move(other)) {}

        virtual ~BinaryCsrView() override {}

        /**
         * Provides read-only access to an individual row in the view.
         */
        typedef const Vector<const uint32> const_row;

        /**
         * Provides access to an individual row in the view and allows to modify it.
         */
        typedef Vector<uint32> row;

        /**
         * Creates and returns a view that provides read-only access to a specific row in the view.
         *
         * @param row   The index of the row
         * @return      An object of type `const_row` that has been created
         */
        const_row operator[](uint32 row) const {
            return Vector<const uint32>(indices_cbegin(row), Matrix::numCols);
        }

        /**
         * Creates and returns a view that provides access to a specific row in the view and allows to modify it.
         *
         * @param row   The index of the row
         * @return      An object of type `row` that has been created
         */
        row operator[](uint32 row) {
            return Vector<uint32>(indices_begin(row), Matrix::numCols);
        }

        /**
         * Returns an `index_const_iterator` to the beginning of the indices in a specific row of the matrix.
         *
         * @param row   The index of the row
         * @return      An `index_const_iterator` to the beginning of the indices
         */
        typename BinarySparseMatrix::index_const_iterator indices_cbegin(uint32 row) const {
            return &BinarySparseMatrix::indices[BinarySparseMatrix::indptr[row]];
        }

        /**
         * Returns an `index_const_iterator` to the end of the indices in a specific row of the matrix.
         *
         * @param row   The index of the row
         * @return      An `index_const_iterator` to the end of the indices
         */
        typename BinarySparseMatrix::index_const_iterator indices_cend(uint32 row) const {
            return &BinarySparseMatrix::indices[BinarySparseMatrix::indptr[row + 1]];
        }

        /**
         * Returns an `index_iterator` to the beginning of the indices in a specific row of the matrix.
         *
         * @param row   The index of the row
         * @return      An `index_iterator` to the beginning of the indices
         */
        typename BinarySparseMatrix::index_iterator indices_begin(uint32 row) {
            return &BinarySparseMatrix::indices[BinarySparseMatrix::indptr[row]];
        }

        /**
         * Returns an `index_iterator` to the end of the indices in a specific row of the matrix.
         *
         * @param row   The index of the row
         * @return      An `index_iterator` to the end of the indices
         */
        typename BinarySparseMatrix::index_iterator indices_end(uint32 row) {
            return &BinarySparseMatrix::indices[BinarySparseMatrix::indptr[row + 1]];
        }

        /**
         * Returns the number of non-zero elements in the view.
         *
         * @return The number of non-zero elements
         */
        uint32 getNumNonZeroElements() const {
            return BinarySparseMatrix::indptr[Matrix::numRows];
        }
};
