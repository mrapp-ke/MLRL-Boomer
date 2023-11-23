/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_matrix_sparse.hpp"

/**
 * A two-dimensional view that provides row-wise access to values stored in a matrix in the compressed sparse row (CSR)
 * format.
 *
 * @tparam T The type of the values, the view provides access to
 */
template<typename T>
class CsrView : public SparseMatrix<T> {
    public:

        /**
         * @param array     A pointer to an array of template type `T` that stores all non-zero values, the view should
         *                  provide access to
         * @param indices   A pointer to an array of type `uint32`, shape `(numNonZeroValues)`, that stores the column
         *                  indices, the values in `array` correspond to
         * @param indptr    A pointer to an array of type `uint32`, shape `(numRows + 1)`, that stores the indices of
         *                  the first element in `array` and `indices` that corresponds to a certain row. The index at
         *                  the last position must be equal to `numNonZeroValues`
         * @param numRows   The number of rows in the view
         * @param numCols   The number of columns in the view
         */
        CsrView(T* array, uint32* indices, uint32* indptr, uint32 numRows, uint32 numCols)
            : SparseMatrix<T>(array, indices, indptr, numRows, numCols) {}

        /**
         * @param other A const reference to an object of type `CsrView` that should be copied
         */
        CsrView(const CsrView<T>& other)
            : SparseMatrix<T>(other.array, other.indices, other.indptr, other.numRows, other.numCols) {}

        /**
         * @param other A reference to an object of type `CsrView` that should be moved
         */
        CsrView(CsrView<T>&& other)
            : SparseMatrix<T>(other.array, other.indices, other.indptr, other.numRows, other.numCols) {}

        virtual ~CsrView() override {}

        /**
         * Returns a `value_const_iterator` to the beginning of the values in a specific row of the matrix.
         *
         * @param row   The index of the row
         * @return      A `value_const_iterator` to the beginning of the values
         */
        typename SparseMatrix<T>::value_const_iterator values_cbegin(uint32 row) const {
            return &View<T>::array[SparseMatrix<T>::indptr[row]];
        }

        /**
         * Returns a `value_const_iterator` to the end of the values in a specific row of the matrix.
         *
         * @param row   The index of the row
         * @return      A `value_const_iterator` to the end of the values
         */
        typename SparseMatrix<T>::value_const_iterator values_cend(uint32 row) const {
            return &View<T>::array[SparseMatrix<T>::indptr[row + 1]];
        }

        /**
         * Returns an `index_const_iterator` to the beginning of the indices in a specific row of the matrix.
         *
         * @param row   The index of the row
         * @return      An `index_const_iterator` to the beginning of the indices
         */
        typename SparseMatrix<T>::index_const_iterator indices_cbegin(uint32 row) const {
            return &SparseMatrix<T>::indices[SparseMatrix<T>::indptr[row]];
        }

        /**
         * Returns an `index_const_iterator` to the end of the indices in a specific row of the matrix.
         *
         * @param row   The index of the row
         * @return      An `index_const_iterator` to the end of the indices
         */
        typename SparseMatrix<T>::index_const_iterator indices_cend(uint32 row) const {
            return &SparseMatrix<T>::indices[SparseMatrix<T>::indptr[row + 1]];
        }

        /**
         * Returns a `value_iterator` to the beginning of the values in a specific row of the matrix.
         *
         * @param row   The index of the row
         * @return      A `value_iterator` to the beginning of the values
         */
        typename SparseMatrix<T>::value_iterator values_begin(uint32 row) {
            return &View<T>::array[SparseMatrix<T>::indptr[row]];
        }

        /**
         * Returns a `value_iterator` to the end of the values in a specific row of the matrix.
         *
         * @param row   The index of the row
         * @return      A `value_iterator` to the end of the values
         */
        typename SparseMatrix<T>::value_iterator values_end(uint32 row) {
            return &View<T>::array[SparseMatrix<T>::indptr[row + 1]];
        }

        /**
         * Returns an `index_iterator` to the beginning of the indices in a specific row of the matrix.
         *
         * @param row   The index of the row
         * @return      An `index_iterator` to the beginning of the indices
         */
        typename SparseMatrix<T>::index_iterator indices_begin(uint32 row) {
            return &SparseMatrix<T>::indices[SparseMatrix<T>::indptr[row]];
        }

        /**
         * Returns an `index_iterator` to the end of the indices in a specific row of the matrix.
         *
         * @param row   The index of the row
         * @return      An `index_iterator` to the end of the indices
         */
        typename SparseMatrix<T>::index_iterator indices_end(uint32 row) {
            return &SparseMatrix<T>::indices[SparseMatrix<T>::indptr[row + 1]];
        }
};
