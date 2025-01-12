/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/regression_matrix_row_wise.hpp"

#include <memory>

/**
 * Defines an interface for all regression matrices that provide row-wise access to the regression scores of individual
 * examples that are stored in a sparse matrix in the compressed sparse row (CSR) format.
 */
class MLRLCOMMON_API ICsrRegressionMatrix : virtual public IRowWiseRegressionMatrix {
    public:

        virtual ~ICsrRegressionMatrix() override {}
};

/**
 * Creates and returns a new object of the type `ICsrRegressionMatrix`.
 *
 * @param values    A pointer to an array of type `float32`, shape `(numDenseElements)`, that stores the values of all
 *                  dense elements explicitly stored in the matrix
 * @param indices   A pointer to an array of type `uint32`, shape `(numDenseElements)`, that stores the column indices
 *                  of all dense elements explicitly stored in the matrix
 * @param indptr    A pointer to an array of type `uint32`, shape `(numRows + 1)`, that stores the indices of the first
 *                  element in `indices` that corresponds to a certain row. The index at the last position is equal to
 *                  `numDenseElements`
 * @param numRows   The number of rows in the regression matrix
 * @param numCols   The number of columns in the regression matrix
 * @return          An unique pointer to an object of type `ICsrRegressionMatrix` that has been created
 */
MLRLCOMMON_API std::unique_ptr<ICsrRegressionMatrix> createCsrRegressionMatrix(float32* values, uint32* indices,
                                                                               uint32* indptr, uint32 numRows,
                                                                               uint32 numCols);
