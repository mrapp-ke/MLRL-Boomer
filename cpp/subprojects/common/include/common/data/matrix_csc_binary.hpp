/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/view_csc_binary.hpp"

/**
 * A two-dimensional matrix that provides column-wise access to binary elements stored in the compressed sparse column
 * (CSC) format.
 */
class BinaryCscMatrix : public BinaryCscView {
    public:

        /**
         * @param numRows               The number of rows in the matrix
         * @param numCols               The number of columns in the matrix
         * @param numNonZeroElements    The number of non-zero elements to be stored by the matrix
         */
        BinaryCscMatrix(uint32 numRows, uint32 numCols, uint32 numNonZeroElements);

        virtual ~BinaryCscMatrix() override;
};
