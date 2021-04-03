/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/matrix_dense.hpp"


namespace seco {

    /**
     * A two-dimensional matrix that stores the weights of individual examples and labels in a C-contiguous array.
     *
     * @tparam T The type of the weights
     */
    template<class T>
    class DenseWeightMatrix final : public DenseMatrix<T> {

        public:

            /**
             * @param numRows   The number of rows in the matrix
             * @param numCols   The number of columns in the matrix
             */
            DenseWeightMatrix(uint32 numRows, uint32 numCols);

    };

}
