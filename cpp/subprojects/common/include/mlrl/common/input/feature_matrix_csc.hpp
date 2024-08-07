/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/feature_matrix_column_wise.hpp"

#include <memory>

/**
 * Defines an interface for all feature matrices that provide column-wise access to the feature values of examples that
 * are stored in a sparse matrix in the compressed sparse column (CSC) format.
 */
class MLRLCOMMON_API ICscFeatureMatrix : public IColumnWiseFeatureMatrix {
    public:

        virtual ~ICscFeatureMatrix() override {}
};

/**
 * Creates and returns a new object of the type `ICscFeatureMatrix`.
 *
 * @param values        A pointer to an array of type `float32`, shape `(numDenseElements)`, that stores the values of
 *                      all dense elements explicitly stored in the matrix
 * @param indices       A pointer to an array of type `uint32`, shape `(numDenseElements)`, that stores the row-indices,
 *                      the values in `values` correspond to
 * @param indptr        A pointer to an array of type `uint32`, shape `(numCols + 1)`, that stores the indices of the
 *                      first element in `values` and `indices` that corresponds to a certain column. The index at the
 *                      last position is equal to `numDenseElements`
 * @param numRows       The number of rows in the feature matrix
 * @param numCols       The number of columns in the feature matrix
 * @param sparseValue   The value that should be used for sparse elements in the feature matrix
 * @return              An unique pointer to an object of type `ICscFeatureMatrix` that has been created
 */
MLRLCOMMON_API std::unique_ptr<ICscFeatureMatrix> createCscFeatureMatrix(const float32* values, uint32* indices,
                                                                         uint32* indptr, uint32 numRows, uint32 numCols,
                                                                         float32 sparseValue = 0.0f);
