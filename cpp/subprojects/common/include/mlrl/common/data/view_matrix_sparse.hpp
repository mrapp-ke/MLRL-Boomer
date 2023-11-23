/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_matrix.hpp"

/**
 * A two-dimensional view that provides row-or column wise access to values stored in a sparse matrix.
 *
 * @tparam T The type of the values, the view provides access to
 */
template<typename T>
class SparseMatrix : public Matrix<T> {
    public:

        /**
         * A pointer to an array that stores the row or column indices, the values in `View::array` correspond to.
         */
        uint32* indices;

        /**
         * A pointer to an array that stores the indices of the first element in `View::array` and `indices` that
         * corresponds to a certain column, if `indices` stores row indices, or row, if `indices` stores column indices.
         */
        uint32* indptr;

        /**
         * @param array     A pointer to an array of template type `T` that stores all non-zero values, the view should
         *                  provide access to
         * @param indices   A pointer to an array of type `uint32`, shape `(numNonZeroValues)`, that stores the row or
         *                  column indices, the values in `array` correspond to
         * @param indptr    A pointer to an array of type `uint32`, shape `(numCols + 1)` or `(numRows + 1)`, that
         *                  stores the indices of the first element in `array` and `indices` that corresponds to a
         *                  certain column, if `indices` stores row indices, or row, if `indices` stores column indices.
         *                  The index at the last position must be equal to `numNonZeroValues`
         * @param numRows   The number of rows in the view
         * @param numCols   The number of columns in the view
         */
        SparseMatrix(T* array, uint32* indices, uint32* indptr, uint32 numRows, uint32 numCols)
            : Matrix<T>(array, numRows, numCols), indices(indices), indptr(indptr) {}

        /**
         * @param other A const reference to an object of type `SparseMatrix` that should be copied
         */
        SparseMatrix(const SparseMatrix<T>& other)
            : Matrix<T>(other.array, other.numRows, other.numCols), indices(other.indices), indptr(other.indptr) {}

        /**
         * @param other A reference to an object of type `SparseMatrix` that should be moved
         */
        SparseMatrix(SparseMatrix<T>&& other)
            : Matrix<T>(other.array, other.numRows, other.numCols), indices(other.indices), indptr(other.indptr) {}

        virtual ~SparseMatrix() override {}
};
