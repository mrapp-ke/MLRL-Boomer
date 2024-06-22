/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_matrix.hpp"

#include <utility>

/**
 * A two-dimensional view that provides row- or column wise access to binary values stored in a sparse matrix.
 */
class MLRLCOMMON_API BinarySparseMatrix : public Matrix {
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
         * @param indices   A pointer to an array of type `uint32`, shape `(numDenseElements)`, that stores the row or
         *                  column indices of all dense elements explicitly stored in the matrix
         * @param indptr    A pointer to an array of type `uint32`, shape `(numCols + 1)` or `(numRows + 1)`, that
         *                  stores the indices of the first element in `indices` that corresponds to a certain column,
         *                  if `indices` stores row indices, or row, if `indices` stores column indices. The index at
         *                  the last position must be equal to `numDenseElements`
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

        /**
         * Releases the ownership of the array that stores the row or column indices, the dense elements explicitly
         * stored in the matrix correspond to. As a result, the behavior of this view becomes undefined and it should
         * not be used anymore. The caller is responsible for freeing the memory that is occupied by the array.
         *
         * @return  A pointer to the array that stores the row or column indices, the dense elements explicitly stored
         *          in the matrix correspond to
         */
        index_type* releaseIndices() {
            index_type* ptr = indices;
            indices = nullptr;
            return ptr;
        }

        /**
         * Releases the ownership of the array that stores the indices of the first dense element that corresponds to
         * a certain column or row. As a result, the behavior of this view becomes undefined and it should not be used
         * anymore. The caller is responsible for freeing the memory that is occupied by the array.
         *
         * @return  A pointer to an array that stores the indices of the first dense element that corresponds to a
         *          certain column or row
         */
        index_type* releaseIndptr() {
            index_type* ptr = indptr;
            indptr = nullptr;
            return ptr;
        }
};

/**
 * Provides row- or column-wise access via iterators to the indices stored in a sparse matrix.
 *
 * @tparam Matrix The type of the matrix
 */
template<typename Matrix>
class MLRLCOMMON_API IterableBinarySparseMatrixDecorator : public Matrix {
    public:

        /**
         * @param view The view, the matrix should be backed by
         */
        explicit IterableBinarySparseMatrixDecorator(typename Matrix::view_type&& view) : Matrix(std::move(view)) {}

        virtual ~IterableBinarySparseMatrixDecorator() override {}

        /**
         * An iterator that provides read-only access to the indices in the matrix.
         */
        typedef typename Matrix::view_type::index_const_iterator index_const_iterator;

        /**
         * An iterator that provides access to the indices in the matrix and allows to modify them.
         */
        typedef typename Matrix::view_type::index_iterator index_iterator;

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

        /**
         * Returns the number of dense elements explicitly stored in the matrix.
         *
         * @return The number of dense elements explicitly stored in the matrix
         */
        uint32 getNumDenseElements() const {
            return Matrix::view.getNumDenseElements();
        }
};
