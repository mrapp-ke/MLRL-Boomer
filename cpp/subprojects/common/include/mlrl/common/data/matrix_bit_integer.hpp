/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_matrix_bit.hpp"

/**
 * A two-dimensional matrix that stores integer values, each with a specific number of bits.
 */
class IntegerBitMatrix final : public MatrixDecorator<AllocatedBitMatrix> {
    public:

        /**
         * @param numRows           The number of rows in the matrix
         * @param numCols           The number of columns in the matrix
         * @param numBitsPerElement The number of bits per element in the matrix
         * @param init              True, if all elements in the matrix should be value-initialized, false otherwise
         */
        IntegerBitMatrix(uint32 numRows, uint32 numCols, uint32 numBitsPerElements, bool init = false)
            : MatrixDecorator<AllocatedBitMatrix>(AllocatedBitMatrix(numRows, numCols, numBitsPerElements, init)) {}
};
