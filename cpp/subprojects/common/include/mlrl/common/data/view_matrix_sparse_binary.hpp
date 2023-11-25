/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_matrix.hpp"

/**
 * A two-dimensional view that provides row-or column wise access to binary values stored in a sparse matrix.
 */
class BinarySparseMatrix : public Matrix {
    public:

        /**
         * A pointer to an array that stores the row or column indices, the values in the matrix correspond to.
         */
        uint32* indices;

        /**
         * A pointer to an array that stores the indices of the first element in `indices` that corresponds to a certain
         * column, if `indices` stores row indices, or row, if `indices` stores column indices.
         */
        uint32* indptr;

        /**
         * @param indices   A pointer to an array of type `uint32`, shape `(numNonZeroValues)`, that stores the row or
         *                  column indices, the values in the matrix correspond to
         * @param indptr    A pointer to an array of type `uint32`, shape `(numCols + 1)` or `(numRows + 1)`, that
         *                  stores the indices of the first element in `indices` that corresponds to a certain column,
         *                  if `indices` stores row indices, or row, if `indices` stores column indices. The index at
         *                  the last position must be equal to `numNonZeroValues`
         * @param numRows   The number of rows in the view
         * @param numCols   The number of columns in the view
         */
        BinarySparseMatrix(uint32* indices, uint32* indptr, uint32 numRows, uint32 numCols)
            : Matrix(numRows, numCols), indices(indices), indptr(indptr) {}

        /**
         * @param other A const reference to an object of type `SparseMatrix` that should be copied
         */
        BinarySparseMatrix(const BinarySparseMatrix& other)
            : Matrix(other), indices(other.indices), indptr(other.indptr) {}

        /**
         * @param other A reference to an object of type `SparseMatrix` that should be moved
         */
        BinarySparseMatrix(BinarySparseMatrix&& other)
            : Matrix(std::move(other)), indices(other.indices), indptr(other.indptr) {}

        virtual ~BinarySparseMatrix() override {}

        /**
         * The type of the indices, the view provides access to.
         */
        typedef uint32 index_type;

        /**
         * An iterator that provides read-only access to the indices in the view.
         */
        typedef View<index_type>::const_iterator index_const_iterator;

        /**
         * An iterator that provides access to the indices in the view and allows to modify them.
         */
        typedef View<index_type>::iterator index_iterator;
};

/**
 * Provides row- or column-wise access via iterators to the indices stored in a sparse matrix.
 *
 * @tparam Matrix The type of the matrix
 */
template<typename Matrix>
class IterableBinarySparseMatrixDecorator : public Matrix {
    public:

        /**
         * @param view The view, the matrix should be backed by
         */
        IterableBinarySparseMatrixDecorator(typename Matrix::view_type&& view) : Matrix(std::move(view)) {}

        virtual ~IterableBinarySparseMatrixDecorator() override {}

        /**
         * An iterator that provides read-only access to the indices in the matrix.
         */
        typedef const uint32* index_const_iterator;

        /**
         * An iterator that provides access to the indices in the matrix and allows to modify them.
         */
        typedef uint32* index_iterator;

        /**
         * Returns an `index_const_iterator` to the beginning of the indices in a specific row or column of the matrix,
         * depending on the memory layout of the view, the matrix is backed by.
         *
         * @param index The index of the row or column
         * @return      An `index_const_iterator` to the beginning of the indices
         */
        index_const_iterator indices_cbegin(uint32 index) const {
            return Matrix::view.indices_cbegin(index);
        }

        /**
         * Returns an `index_const_iterator` to the end of the indices in a specific row or column of the matrix,
         * depending on the memory layout of the view, the matrix is backed by.
         *
         * @param index The index of the row or column
         * @return      An `index_const_iterator` to the end of the indices
         */
        index_const_iterator indices_cend(uint32 index) const {
            return Matrix::view.indices_cend(index);
        }

        /**
         * Returns an `index_iterator` to the beginning of the indices in a specific row or column of the matrix,
         * depending on the memory layout of the view, the matrix is backed by.
         *
         * @param index The index of the row or column
         * @return      An `index_iterator` to the beginning of the indices
         */
        index_iterator indices_begin(uint32 index) {
            return Matrix::view.indices_begin(index);
        }

        /**
         * Returns an `index_iterator` to the end of the indices in a specific row or column of the matrix, depending on
         * the memory layout of the view, the matrix is backed by.
         *
         * @param index The index of the row or column
         * @return      An `index_iterator` to the end of the indices
         */
        index_iterator indices_end(uint32 index) {
            return Matrix::view.indices_end(index);
        }
};
