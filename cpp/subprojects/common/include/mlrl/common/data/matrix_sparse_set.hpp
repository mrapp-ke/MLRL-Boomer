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
using SparseSetMatrixDecorator = IterableSparseSetViewDecorator<MatrixDecorator<Matrix>>;

/**
 * A two-dimensional matrix that provides random read and write access, as well as row-wise read and write access via
 * iterators, to values stored in a sparse matrix in the list of lists (LIL) format.
 *
 * @tparam T The type of the values that are stored in the matrix
 */
template<typename T>
class SparseSetMatrix final : public SparseSetMatrixDecorator<AllocatedSparseSetView<T>> {
    public:

        /**
         * @param numRows   The number of rows in the matrix
         * @param numCols   The number of columns in the matrix
         */
        SparseSetMatrix(uint32 numRows, uint32 numCols)
            : SparseSetMatrixDecorator<AllocatedSparseSetView<T>>(AllocatedSparseSetView<T>(numRows, numCols)) {}
};
