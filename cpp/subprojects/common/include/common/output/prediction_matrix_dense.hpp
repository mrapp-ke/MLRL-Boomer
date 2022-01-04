/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/view_c_contiguous.hpp"


/**
 * A dense matrix that provides read-only access to predictions that are stored in a C-contiguous array.
 *
 * @tparam T The type of the predictions that are stored by the matrix
 */
template<typename T>
class DensePredictionMatrix final : public CContiguousConstView<T> {

    private:

        T* matrix_;

    public:

        /**
         * @param numRows   The number of rows in the matrix
         * @param numCols   The number of columns in the matrix
         */
        DensePredictionMatrix(uint32 numRows, uint32 numCols);

        ~DensePredictionMatrix();

        /**
         * Releases the ownership of the array that stores the predictions. The caller is responsible for freeing the
         * memory that is occupied by the array.
         *
         * @return A pointer to the array that stores the predictions
         */
        T* release();

};
