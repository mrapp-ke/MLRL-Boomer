/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_matrix_sparse_binary.hpp"

/**
 * A two-dimensional view that provides row-or column wise access to values stored in a sparse matrix.
 *
 * @tparam T The type of the values, the view provides access to
 */
template<typename T>
class SparseMatrix : public BinarySparseMatrix {
    public:

        /**
         * A pointer to the array that stores the values, the view provides access to.
         */
        T* values;

        /**
         * @param values    A pointer to an array of template type `T` that stores all non-zero values, the view should
         *                  provide access to
         * @param indices   A pointer to an array of type `uint32`, shape `(numNonZeroValues)`, that stores the row or
         *                  column indices, the values in `values` correspond to
         * @param indptr    A pointer to an array of type `uint32`, shape `(numCols + 1)` or `(numRows + 1)`, that
         *                  stores the indices of the first element in `values` and `indices` that corresponds to a
         *                  certain column, if `indices` stores row indices, or row, if `indices` stores column indices.
         *                  The index at the last position must be equal to `numNonZeroValues`
         * @param numRows   The number of rows in the view
         * @param numCols   The number of columns in the view
         */
        SparseMatrix(T* values, uint32* indices, uint32* indptr, uint32 numRows, uint32 numCols)
            : BinarySparseMatrix(numRows, numCols, indices, indptr), values(values) {}

        /**
         * @param other A const reference to an object of type `SparseMatrix` that should be copied
         */
        SparseMatrix(const SparseMatrix<T>& other) : BinarySparseMatrix(other), values(other.values) {}

        /**
         * @param other A reference to an object of type `SparseMatrix` that should be moved
         */
        SparseMatrix(SparseMatrix<T>&& other) : BinarySparseMatrix(std::move(other)), values(other.values) {}

        virtual ~SparseMatrix() override {}

        /**
         * The type of the values, the view provides access to.
         */
        typedef T value_type;

        /**
         * An iterator that provides read-only access to the values in the view.
         */
        typedef typename View<value_type>::const_iterator value_const_iterator;

        /**
         * An iterator that provides access to the values in the view and allows to modify them.
         */
        typedef typename View<value_type>::iterator value_iterator;
};

/**
 * Provides row- or column-wise access via iterators to the indices and values stored in a sparse matrix.
 *
 * @tparam Matrix The type of the matrix
 */
template<typename Matrix>
class IterableSparseMatrixDecorator : public IterableBinarySparseMatrixDecorator<Matrix> {
    public:

        /**
         * @param view The view, the matrix should be backed by
         */
        IterableSparseMatrixDecorator(typename Matrix::view_type&& view)
            : IterableBinarySparseMatrixDecorator<Matrix>(std::move(view)) {}

        virtual ~IterableSparseMatrixDecorator() override {}

        /**
         * An iterator that provides read-only access to the values in the matrix.
         */
        typedef typename Matrix::view_type::const_iterator value_const_iterator;

        /**
         * An iterator that provides access to the values in the matrix and allows to modify them.
         */
        typedef typename Matrix::view_type::iterator value_iterator;

        /**
         * Returns a `value_const_iterator` to the beginning of the values in a specific row or column of the matrix,
         * depending on the memory layout of the view, the matrix is backed by.
         *
         * @param index The index of the row or column
         * @return      A `value_const_iterator` to the beginning of the values
         */
        value_const_iterator values_cbegin(uint32 index) const {
            return Matrix::view.values_cbegin(index);
        }

        /**
         * Returns a `value_const_iterator` to the end of the values in a specific row or column of the matrix,
         * depending on the memory layout of the view, the matrix is backed by.
         *
         * @param index The index of the row or column
         * @return      A `value_const_iterator` to the end of the values
         */
        value_const_iterator values_cend(uint32 index) const {
            return Matrix::view.values_cend(index);
        }

        /**
         * Returns a `value_iterator` to the beginning of the values in a specific row or column of the matrix,
         * depending on the memory layout of the view, the matrix is backed by.
         *
         * @param index The index of the row or column
         * @return      A `value_iterator` to the beginning of the values
         */
        value_iterator values_begin(uint32 index) {
            return Matrix::view.values_begin(index);
        }

        /**
         * Returns a `value_iterator` to the end of the values in a specific row or column of the matrix, depending on
         * the memory layout of the view, the matrix is backed by.
         *
         * @param index The index of the row or column
         * @return      A `value_iterator` to the end of the values
         */
        value_iterator values_end(uint32 index) {
            return Matrix::view.values_end(index);
        }
};
