/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_matrix_sparse.hpp"

#include <utility>

/**
 * A two-dimensional view that provides column-wise access to values stored in a matrix in the compressed sparse column
 * (CSC) format.
 *
 * @tparam T The type of the values, the view provides access to
 */
template<typename T>
class MLRLCOMMON_API CscView : public SparseMatrix<T> {
    public:

        /**
         * @param values        A pointer to an array of template type `T` that stores the values of all dense elements
         *                      explicitly stored in the view
         * @param indices       A pointer to an array of type `uint32`, shape `(numDenseElements)`, that stores the row
         *                      indices, the values in `values` correspond to
         * @param indptr        A pointer to an array of type `uint32`, shape `(numCols + 1)`, that stores the indices
         *                      of the first element in `values` and `indices` that corresponds to a certain column. The
         *                      index at the last position must be equal to `numDenseElements`
         * @param numRows       The number of rows in the view
         * @param numCols       The number of columns in the view
         * @param sparseValue   The value that should be used for sparse elements in the matrix
         */
        CscView(T* values, uint32* indices, uint32* indptr, uint32 numRows, uint32 numCols, T sparseValue = 0)
            : SparseMatrix<T>(values, indices, indptr, numRows, numCols, sparseValue) {}

        /**
         * @param other A const reference to an object of type `CscView` that should be copied
         */
        CscView(const CscView<T>& other) : SparseMatrix<T>(other) {}

        /**
         * @param other A reference to an object of type `CscView` that should be moved
         */
        CscView(CscView<T>&& other) : SparseMatrix<T>(std::move(other)) {}

        virtual ~CscView() override {}

        /**
         * Returns a `value_const_iterator` to the beginning of the values in a specific column of the matrix.
         *
         * @param column    The index of the column
         * @return          A `value_const_iterator` to the beginning of the values
         */
        typename SparseMatrix<T>::value_const_iterator values_cbegin(uint32 column) const {
            return &SparseMatrix<T>::values[SparseMatrix<T>::indptr[column]];
        }

        /**
         * Returns a `value_const_iterator` to the end of the values in a specific column of the matrix.
         *
         * @param column    The index of the column
         * @return          A `value_const_iterator` to the end of the values
         */
        typename SparseMatrix<T>::value_const_iterator values_cend(uint32 column) const {
            return &SparseMatrix<T>::values[SparseMatrix<T>::indptr[column + 1]];
        }

        /**
         * Returns an `index_const_iterator` to the beginning of the indices in a specific column of the matrix.
         *
         * @param column    The index of the column
         * @return          An `index_const_iterator` to the beginning of the indices
         */
        typename SparseMatrix<T>::index_const_iterator indices_cbegin(uint32 column) const {
            return &SparseMatrix<T>::indices[SparseMatrix<T>::indptr[column]];
        }

        /**
         * Returns an `index_const_iterator` to the end of the indices in a specific column of the matrix.
         *
         * @param column    The index of the column
         * @return          An `index_const_iterator` to the end of the indices
         */
        typename SparseMatrix<T>::index_const_iterator indices_cend(uint32 column) const {
            return &SparseMatrix<T>::indices[SparseMatrix<T>::indptr[column + 1]];
        }

        /**
         * Returns a `value_iterator` to the beginning of the values in a specific column of the matrix.
         *
         * @param column    The index of the column
         * @return          A `value_iterator` to the beginning of the values
         */
        typename SparseMatrix<T>::value_iterator values_begin(uint32 column) {
            return &SparseMatrix<T>::values[SparseMatrix<T>::indptr[column]];
        }

        /**
         * Returns a `value_iterator` to the end of the values in a specific column of the matrix.
         *
         * @param column    The index of the column
         * @return          A `value_iterator` to the end of the values
         */
        typename SparseMatrix<T>::value_iterator values_end(uint32 column) {
            return &SparseMatrix<T>::values[SparseMatrix<T>::indptr[column + 1]];
        }

        /**
         * Returns an `index_iterator` to the beginning of the indices in a specific column of the matrix.
         *
         * @param column    The index of the column
         * @return          An `index_iterator` to the beginning of the indices
         */
        typename SparseMatrix<T>::index_iterator indices_begin(uint32 column) {
            return &SparseMatrix<T>::indices[SparseMatrix<T>::indptr[column]];
        }

        /**
         * Returns an `index_iterator` to the end of the indices in a specific column of the matrix.
         *
         * @param column    The index of the column
         * @return          An `index_iterator` to the end of the indices
         */
        typename SparseMatrix<T>::index_iterator indices_end(uint32 column) {
            return &SparseMatrix<T>::indices[SparseMatrix<T>::indptr[column + 1]];
        }

        /**
         * Returns the number of dense elements explicitly stored in the view.
         *
         * @return The number of dense elements explicitly stored in the view
         */
        uint32 getNumDenseElements() const {
            return SparseMatrix<T>::indptr[Matrix::numCols];
        }
};
