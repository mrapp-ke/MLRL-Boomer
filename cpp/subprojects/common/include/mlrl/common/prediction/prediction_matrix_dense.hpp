/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/matrix_dense.hpp"
#include "mlrl/common/data/view_matrix_c_contiguous.hpp"
#include "mlrl/common/util/dll_exports.hpp"

/**
 * A dense matrix that provides read-only access to predictions that are stored in a C-contiguous array.
 *
 * @tparam T The type of the predictions that are stored by the matrix
 */
template<typename T>
class MLRLCOMMON_API DensePredictionMatrix final : public DenseMatrixDecorator<AllocatedCContiguousView<T>> {
    public:

        /**
         * @param numRows   The number of rows in the matrix
         * @param numCols   The number of columns in the matrix
         * @param init      True, if all elements in the matrix should be value-initialized, false otherwise
         */
        DensePredictionMatrix(uint32 numRows, uint32 numCols, bool init = false);

        /**
         * Returns a pointer to the array that stores the predictions.
         *
         * @return A pointer to the array that stores the predictions
         */
        T* get();

        /**
         * Releases the ownership of the array that stores the predictions. As a result, the behavior of this matrix
         * becomes undefined and it should not be used anymore. The caller is responsible for freeing the memory that is
         * occupied by the array.
         *
         * @return A pointer to the array that stores the predictions
         */
        T* release();
};
