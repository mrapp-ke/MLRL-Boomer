/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/indexed_value.hpp"
#include "mlrl/common/data/view_matrix_lil.hpp"

/**
 * Provides row-wise read and write access via iterators to the values stored in a sparse matrix in the list of lists
 * (LIL) format.
 *
 * @tparam Matrix The type of the matrix
 */
template<typename Matrix>
using LilMatrixDecorator = IterableListOfListsDecorator<MatrixDecorator<Matrix>>;

/**
 * A two-dimensional matrix that provides row-wise read and write access to values that are stored in a newly allocated
 * sparse matrix in the list of lists (LIL) format.
 */
template<typename T>
class LilMatrix final : public LilMatrixDecorator<AllocatedListOfLists<IndexedValue<T>>> {
    public:

        /**
         * @param numRows   The number of rows in the matrix
         * @param numCols   The number of columns in the matrix
         */
        LilMatrix(uint32 numRows, uint32 numCols)
            : LilMatrixDecorator<AllocatedListOfLists<IndexedValue<T>>>(
                AllocatedListOfLists<IndexedValue<T>>(numRows, numCols)) {}
};
