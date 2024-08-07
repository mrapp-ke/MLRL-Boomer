/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/regression_matrix_row_wise.hpp"

#include <memory>

/**
 * Defines an interface for all regression matrices that provide row-wise access to the regression scores of individual
 * examples that are stored in a C-contiguous array.
 */
class MLRLCOMMON_API ICContiguousRegressionMatrix : public IRowWiseRegressionMatrix {
    public:

        virtual ~ICContiguousRegressionMatrix() override {}
};

/**
 * Creates and returns a new object of the type `ICContiguousRegressionMatrix`.
 *
 * @param array     A pointer to a C-contiguous array of type `float32` that stores the regression scores
 * @param numRows   The number of rows in the regression matrix
 * @param numCols   The number of columns in the regression matrix
 * @return          An unique pointer to an object of type `ICContiguousRegressionMatrix` that has been created
 */
MLRLCOMMON_API std::unique_ptr<ICContiguousRegressionMatrix> createCContiguousRegressionMatrix(const float32* array,
                                                                                               uint32 numRows,
                                                                                               uint32 numCols);
