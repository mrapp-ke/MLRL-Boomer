/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_matrix_sparse_set.hpp"

/**
 * Provides random read and write access, as well as row-wise read and write access via iterators, to values stored in a
 * sparse matrix in the list of lists (LIL) format.
 *
 * @tparam Matrix The type of the matrix
 */
template<typename Matrix>
using SparseSetMatrixDecorator = MatrixDecorator<Matrix>;

/**
 * A two-dimensional matrix that provides row-wise access to data that is stored in the list of lists (LIL) format. In
 * contrast to a `LilMatrix`, this matrix does also provide random access to its elements. This additional functionality
 * comes at the expense of memory efficiency, as it requires to not only maintain a sparse matrix that stores the
 * non-zero elements, but also a dense matrix that stores for each element the corresponding position in the sparse
 * matrix, if available.
 *
 * The data structure that is used for the representation of a single row is often referred to as an "unordered sparse
 * set". It was originally proposed in "An efficient representation for sparse sets", Briggs, Torczon, 1993 (see
 * https://dl.acm.org/doi/pdf/10.1145/176454.176484).
 *
 * @tparam T The type of the values that are stored in the matrix
 */
template<typename T>
class SparseSetMatrix : public SparseSetMatrixDecorator<SparseSetView<T>> {
    public:

        /**
         * @param numRows   The number of rows in the matrix
         * @param numCols   The number of columns in the matrix
         */
        SparseSetMatrix(uint32 numRows, uint32 numCols);

        virtual ~SparseSetMatrix() {}

        /**
         * Provides access to a row and allows to modify its elements.
         */
        typedef typename SparseSetView<T>::row row;

        /**
         * Provides read-only access to a row.
         */
        typedef typename SparseSetView<T>::const_row const_row;

        /**
         * An iterator that provides access to the elements at a row and allows to modify them.
         */
        typedef typename SparseSetView<T>::iterator iterator;

        /**
         * An iterator that provides read-only access to the elements at a row.
         */
        typedef typename SparseSetView<T>::const_iterator const_iterator;

        /**
         * Returns an `iterator` to the beginning of a specific row.
         *
         * @param row   The row
         * @return      An `iterator` to the beginning
         */
        iterator begin(uint32 row);

        /**
         * Returns an `iterator` to the end of a specific row.
         *
         * @param row   The row
         * @return      An `iterator` to the end
         */
        iterator end(uint32 row);

        /**
         * Returns a `const_iterator` to the beginning of a specific row.
         *
         * @param row   The row
         * @return      A `const_iterator` to the beginning
         */
        const_iterator cbegin(uint32 row) const;

        /**
         * Returns a `const_iterator` to the end of a specific row.
         *
         * @param row   The row
         * @return      A `const_iterator` to the end
         */
        const_iterator cend(uint32 row) const;

        /**
         * Provides access to a specific row and allows to modify its elements.
         *
         * @param row   The index of the row
         * @return      A `row`
         */
        row operator[](uint32 row);

        /**
         * Provides read-only access to a specific row.
         *
         * @param row   The index of the row
         * @return      A `const_row`
         */
        const_row operator[](uint32 row) const;

        /**
         * Sets the values of all elements to zero.
         */
        void clear();

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
};
