/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/matrix_lil.hpp"

/**
 * A two-dimensional matrix that provides row-wise read and write access to binary values that are stored in a newly
 * allocated sparse matrix in the list of lists (LIL) format.
 */
class BinaryLilMatrix final : public LilMatrixDecorator<AllocatedListOfLists<uint32>> {
    public:

        /**
         * @param numRows   The number of rows in the matrix
         * @param numCols   The number of columns in the matrix
         */
        BinaryLilMatrix(uint32 numRows, uint32 numCols)
            : LilMatrixDecorator<AllocatedListOfLists<uint32>>(AllocatedListOfLists<uint32>(numRows, numCols)) {}
};
