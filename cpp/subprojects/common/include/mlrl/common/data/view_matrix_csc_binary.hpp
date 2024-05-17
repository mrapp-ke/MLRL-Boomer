/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_matrix_sparse.hpp"
#include "mlrl/common/data/view_vector.hpp"

/**
 * A two-dimensional view that provides column-wise access to binary values stored in a matrix in the compressed sparse
 * column (CSC) format.
 */
class MLRLCOMMON_API BinaryCscView : public BinarySparseMatrix {
    public:

        /**
         * True, if non-zero values are associated with sparse elements in the matrix instead of dense ones, false
         * otherwise.
         */
        bool sparseValue;

        /**
         * @param indices       A pointer to an array of type `uint32`, shape `(numDenseElements)`, that stores the row
         *                      indices of all dense elements explicitly stored in the matrix
         * @param indptr        A pointer to an array of type `uint32`, shape `(numCols + 1)`, that stores the indices
         *                      of the first element in `indices` that corresponds to a certain column. The index at the
         *                      position must be equal to `numDenseElements`
         * @param numRows       The number of rows in the view
         * @param numCols       The number of columns in the view
         * @param sparseValue   True, if non-zero values should be associated with sparse elements in the matrix instead
         *                      of dense ones, false otherwise
         */
        BinaryCscView(uint32* indices, uint32* indptr, uint32 numRows, uint32 numCols, bool sparseValue = false)
            : BinarySparseMatrix(indices, indptr, numRows, numCols), sparseValue(sparseValue) {}

        /**
         * @param other A const reference to an object of type `BinaryCscView` that should be copied
         */
        BinaryCscView(const BinaryCscView& other) : BinarySparseMatrix(other), sparseValue(other.sparseValue) {}

        /**
         * @param other A reference to an object of type `BinaryCscView` that should be moved
         */
        BinaryCscView(BinaryCscView&& other) : BinarySparseMatrix(std::move(other)), sparseValue(other.sparseValue) {}

        virtual ~BinaryCscView() override {}

        /**
         * Provides read-only access to an individual column in the view.
         */
        typedef const Vector<const uint32> const_column;

        /**
         * Provides access to an individual column in the view and allows to modify it.
         */
        typedef Vector<uint32> column;

        /**
         * Creates and returns a view that provides read-only access to a specific column in the view.
         *
         * @param column    The index of the column
         * @return          An object of type `const_column` that has been created
         */
        const_column operator[](uint32 column) const {
            return Vector<const uint32>(indices_cbegin(column), Matrix::numRows);
        }

        /**
         * Creates and returns a view that provides access to a specific column in the view and allows to modify it.
         *
         * @param column    The index of the column
         * @return          An object of type `column` that has been created
         */
        column operator[](uint32 column) {
            return Vector<uint32>(indices_begin(column), Matrix::numRows);
        }

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

        /**
         * Returns the number of dense elements explicitly stored in the view.
         *
         * @return The number of dense elements explicitly stored in the view
         */
        uint32 getNumDenseElements() const {
            return BinarySparseMatrix::indptr[Matrix::numCols];
        }
};

/**
 * Allocates the memory for a two-dimensional view that provides column-wise access to binary values stored in a matrix
 * in the compressed sparse column (CSC) format.
 *
 * @tparam Matrix The type of the view
 */
template<typename Matrix>
class MLRLCOMMON_API BinaryCscViewAllocator : public Matrix {
    public:

        /**
         * @param numDenseElements  The number of dense elements explicitly stored in the view
         * @param numRows           The number of rows in the view
         * @param numCols           The number of columns in the view
         * @param sparseValue       True, if non-zero values should be associated with sparse elements in the matrix
         *                          instead of dense ones, false otherwise
         */
        BinaryCscViewAllocator(uint32 numDenseElements, uint32 numRows, uint32 numCols, bool sparseValue = false)
            : Matrix(allocateMemory<uint32>(numDenseElements), allocateMemory<uint32>(numCols + 1), numRows, numCols,
                     sparseValue) {
            Matrix::indptr[0] = 0;
            Matrix::indptr[numCols] = numDenseElements;
        }

        /**
         * @param other A reference to an object of type `BinaryCscViewAllocator` that should be copied
         */
        BinaryCscViewAllocator(const BinaryCscViewAllocator<Matrix>& other) : Matrix(other) {
            throw std::runtime_error("Objects of type BinaryCscViewAllocator cannot be copied");
        }

        /**
         * @param other A reference to an object of type `BinaryCscViewAllocator` that should be moved
         */
        BinaryCscViewAllocator(BinaryCscViewAllocator<Matrix>&& other) : Matrix(std::move(other)) {
            other.releaseIndices();
            other.releaseIndptr();
        }

        virtual ~BinaryCscViewAllocator() override {
            freeMemory(Matrix::indices);
            freeMemory(Matrix::indptr);
        }
};

/**
 * Allocates the memory, a `BinaryCscView` provides access to
 */
typedef BinaryCscViewAllocator<BinaryCscView> AllocatedBinaryCscView;
