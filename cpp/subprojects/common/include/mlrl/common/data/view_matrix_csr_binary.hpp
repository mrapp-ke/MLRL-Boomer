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
class MLRLCOMMON_API BinaryCsrView : public BinarySparseMatrix {
    public:

        /**
         * True, if non-zero values are associated with sparse elements in the matrix instead of dense ones, false
         * otherwise.
         */
        bool sparseValue;

        /**
         * @param indices       A pointer to an array of type `uint32`, shape `(numDenseElements)`, that stores the
         *                      column indices of all dense elements explicitly stored in the matrix
         * @param indptr        A pointer to an array of type `uint32`, shape `(numRows + 1)`, that stores the indices
         *                      of the first element in `indices` that corresponds to a certain row. The index at the
         *                      last position musts be equal to `numDenseElements`
         * @param numRows       The number of rows in the view
         * @param numCols       The number of columns in the view
         * @param sparseValue   True, if non-zero values should be associated with sparse elements in the matrix instead
         *                      of dense ones, false otherwise
         */
        BinaryCsrView(uint32* indices, uint32* indptr, uint32 numRows, uint32 numCols, bool sparseValue = false)
            : BinarySparseMatrix(indices, indptr, numRows, numCols), sparseValue(sparseValue) {}

        /**
         * @param other A const reference to an object of type `BinaryCsrView` that should be copied
         */
        BinaryCsrView(const BinaryCsrView& other) : BinarySparseMatrix(other), sparseValue(other.sparseValue) {}

        /**
         * @param other A reference to an object of type `BinaryCsrView` that should be moved
         */
        BinaryCsrView(BinaryCsrView&& other) : BinarySparseMatrix(std::move(other)), sparseValue(other.sparseValue) {}

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
         * Returns the number of dense elements explicitly stored in the view.
         *
         * @return The number of dense elements explicitly stored in the view
         */
        uint32 getNumDenseElements() const {
            return BinarySparseMatrix::indptr[Matrix::numRows];
        }
};

/**
 * Allocates the memory for a two-dimensional view that provides row-wise access to binary values stored in a matrix in
 * the compressed sparse row (CSR) format.
 *
 * @tparam Matrix The type of the view
 */
template<typename Matrix>
class MLRLCOMMON_API BinaryCsrViewAllocator : public Matrix {
    public:

        /**
         * @param numDenseElements  The number of dense elements explicitly stored in the view
         * @param numRows           The number of rows in the view
         * @param numCols           The number of columns in the view
         * @param sparseValue       True, if non-zero values should be associated with sparse elements in the matrix
         *                          instead of dense ones, false otherwise
         */
        BinaryCsrViewAllocator(uint32 numDenseElements, uint32 numRows, uint32 numCols, bool sparseValue = false)
            : Matrix(allocateMemory<uint32>(numDenseElements), allocateMemory<uint32>(numRows + 1), numRows, numCols,
                     sparseValue) {
            Matrix::indptr[0] = 0;
            Matrix::indptr[numRows] = numDenseElements;
        }

        /**
         * @param other A reference to an object of type `BinaryCsrViewAllocator` that should be copied
         */
        BinaryCsrViewAllocator(const BinaryCsrViewAllocator<Matrix>& other) : Matrix(other) {
            throw std::runtime_error("Objects of type BinaryCsrViewAllocator cannot be copied");
        }

        /**
         * @param other A reference to an object of type `BinaryCsrViewAllocator` that should be moved
         */
        BinaryCsrViewAllocator(BinaryCsrViewAllocator<Matrix>&& other) : Matrix(std::move(other)) {
            other.releaseIndices();
            other.releaseIndptr();
        }

        virtual ~BinaryCsrViewAllocator() override {
            freeMemory(Matrix::indices);
            freeMemory(Matrix::indptr);
        }
};

/**
 * Allocates the memory, a `BinaryCsrView` provides access to
 */
typedef BinaryCsrViewAllocator<BinaryCsrView> AllocatedBinaryCsrView;
