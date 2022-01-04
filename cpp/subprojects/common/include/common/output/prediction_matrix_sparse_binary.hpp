/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/matrix_lil_binary.hpp"
#include "common/data/view_csr_binary.hpp"
#include <memory>


/**
 * A sparse matrix that provides read-only access to binary predictions that are stored in the compressed sparse row
 * (CSR) format.
 *
 * The matrix maintains two arrays, referred to as `rowIndices_` and `colIndices_`. The latter stores a column-index for
 * each of the `numNonZeroValues` non-zero elements in the matrix. The former stores `numRows + 1` row-indices that
 * specify the first element in `colIndices_` that correspond to a certain row. The index at the last position is equal
 * to the number of non-zero values in the matrix.
 */
class BinarySparsePredictionMatrix final {

    private:

        uint32* rowIndices_;

        uint32* colIndices_;

        BinaryCsrConstView view_;

    public:

        /**
         * @param numRows       The number of rows in the matrix
         * @param numCols       The number of columns in the matrix
         * @param rowIndices    A pointer to an array of type `uint32`, shape `(numRows + 1)`, that stores the indices
         *                      of the first element in `colIndices` that corresponds to a certain row. The index at the
         *                      last position is equal to `numNonZeroValues`
         * @param colIndices    A pointer to an array of type `uint32`, shape `(numNonZeroValues)`, that stores the
         *                      column-indices, the non-zero elements correspond to
         */
        BinarySparsePredictionMatrix(uint32 numRows, uint32 numCols, uint32* rowIndices, uint32* colIndices);

        ~BinarySparsePredictionMatrix();

        /**
         * An iterator that provides read-only access to the indices in the matrix.
         */
        typedef BinaryCsrConstView::index_const_iterator index_const_iterator;

        /**
         * An iterator that provides read-only access to the values in the matrix.
         */
        typedef BinaryCsrConstView::value_const_iterator value_const_iterator;

        /**
         * Returns an `index_const_iterator` to the beginning of the indices at a specific row.
         *
         * @param row   The row
         * @return      An `index_const_iterator` to the beginning of the indices
         */
        index_const_iterator row_indices_cbegin(uint32 row) const;

        /**
         * Returns an `index_const_iterator` to the end of the indices at a specific row.
         *
         * @param row   The row
         * @return      An `index_const_iterator` to the end of the indices
         */
        index_const_iterator row_indices_cend(uint32 row) const;

        /**
         * Returns a `value_const_iterator` to the beginning of the values at a specific row.
         *
         * @param row   The row
         * @return      A `value_const_iterator` to the beginning of the values
         */
        value_const_iterator row_values_cbegin(uint32 row) const;

        /**
         * Returns a `value_const_iterator` to the end of the values at a specific row.
         *
         * @param row   The row
         * @return      A `value_const_iterator` to the end of the values
         */
        value_const_iterator row_values_cend(uint32 row) const;

        /**
         * Returns the number of rows in the matrix.
         *
         * @return The number of rows
         */
        uint32 getNumRows() const;

        /**
         * Returns the number of columns in the matrix.
         *
         * @return The number of columns
         */
        uint32 getNumCols() const;

        /**
         * Returns the number of non-zero elements in the matrix.
         *
         * @return The number of non-zero elements
         */
        uint32 getNumNonZeroElements() const;

        /**
         * Releases the ownership of the array `rowIndices_`. The caller is responsible for freeing the memory that is
         * occupied by the array.
         *
         * @return A pointer to the array `rowIndices_`
         */
        uint32* releaseRowIndices();

        /**
         * Releases the ownership of the array `colIndices_`. The caller is responsible for freeing the memory that is
         * occupied by the array.
         *
         * @return A pointer to the array `colIndices_`
         */
        uint32* releaseColIndices();

};

/**
 * Creates and returns a new object of the type `BinarySparsePredictionMatrix` as a copy of an existing
 * `BinaryLilMatrix`.
 *
 * @param lilMatrix             A reference to an object of type `BinaryLilMatrix` to be copied
 * @param numCols               The number of columns of the given `BinaryLilMatrix`
 * @param numNonZeroElements    The number of non-zero elements in the given `BinaryLilMatrix`
 * @return                      An unique pointer to an object of type `BinarySparsePredictionMatrix` that has been
 *                              created
 */
std::unique_ptr<BinarySparsePredictionMatrix> createBinarySparsePredictionMatrix(const BinaryLilMatrix& lilMatrix,
                                                                                 uint32 numCols,
                                                                                 uint32 numNonZeroElements);
