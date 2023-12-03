/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/matrix_dense.hpp"
#include "mlrl/common/data/view_matrix_c_contiguous.hpp"

/**
 * A two-dimensional dense matrix that provides row-wise read and write access via iterators to the values stores in a
 * newly allocated C-contiguous matrix.
 *
 * @tparam T The type of the data that is stored in the matrix
 */
template<typename T>
class CContiguousMatrix final : public DenseMatrixDecorator<AllocatedCContiguousView<T>> {
    public:

        /**
         * @param numRows   The number of rows in the matrix
         * @param numCols   The number of columns in the matrix
         * @param init      True, if all elements in the matrix should be value-initialized, false otherwise
         */
        CContiguousMatrix(uint32 numRows, uint32 numCols, bool init = false)
            : DenseMatrixDecorator<AllocatedCContiguousView<T>>(AllocatedCContiguousView<T>(numRows, numCols, init)) {}
};
