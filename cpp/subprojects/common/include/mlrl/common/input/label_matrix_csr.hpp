/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/label_matrix_row_wise.hpp"

#include <memory>

/**
 * Defines an interface for all label matrices that provide row-wise access to the labels of individual examples that
 * are stored in a sparse matrix in the compressed sparse row (CSR) format.
 */
class MLRLCOMMON_API ICsrLabelMatrix : virtual public IRowWiseLabelMatrix {
    public:

        virtual ~ICsrLabelMatrix() override {}
};

/**
 * Creates and returns a new object of the type `ICsrLabelMatrix`.
 *
 * @param indices   A pointer to an array of type `uint32`, shape `(numDenseElements)`, that stores the column indices
 *                  of all dense elements explicitly stored in the matrix
 * @param indptr    A pointer to an array of type `uint32`, shape `(numRows + 1)`, that stores the indices of the first
 *                  element in `indices` that corresponds to a certain row. The index at the last position is equal to
 *                  `numDenseElements`
 * @param numRows   The number of rows in the label matrix
 * @param numCols   The number of columns in the label matrix
 * @return          An unique pointer to an object of type `ICsrLabelMatrix` that has been created
 */
MLRLCOMMON_API std::unique_ptr<ICsrLabelMatrix> createCsrLabelMatrix(uint32* indices, uint32* indptr, uint32 numRows,
                                                                     uint32 numCols);
