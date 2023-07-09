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

        /**
         * Returns an `index_const_iterator` to the beginning of the array that stores the indices of the first element
         * in `rowIndices` that corresponds to a certain column.
         *
         * @return An `index_const_iterator` to the beginning
         */
        index_const_iterator indptr_cbegin() const;

        /**
         * Returns an `index_const_iterator` to the end of the array that stores the indices of the first element in
         * `rowIndices` that corresponds to a certain column.
         *
         * @return An `index_const_iterator` to the end
         */
        index_const_iterator indptr_cend() const;

        /**
         * Returns an `index_iterator` to the beginning of the array that stores the indices of the first element in
         * `rowIndices` that corresponds to a certain column.
         *
         * @return An `index_iterator` to the beginning
         */
        index_iterator indptr_begin();

        /**
         * Returns an `index_iterator` to the end of the array that stores the indices of the first element in
         * `rowIndices` that corresponds to a certain column.
         *
         * @return An `index_iterator` to the end
         */
        index_iterator indptr_end();
};
