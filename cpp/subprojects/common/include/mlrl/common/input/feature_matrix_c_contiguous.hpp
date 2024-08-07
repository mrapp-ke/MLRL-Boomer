/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/feature_matrix_row_wise.hpp"

#include <memory>

/**
 * Defines an interface for all feature matrices that provide row-wise access to the feature values of examples that are
 * stored in a C-contiguous array.
 */
class MLRLCOMMON_API ICContiguousFeatureMatrix : public IRowWiseFeatureMatrix {
    public:

        virtual ~ICContiguousFeatureMatrix() override {}
};

/**
 * Creates and returns a new object of the type `ICContiguousFeatureMatrix`.
 *
 * @param array     A pointer to a C-contiguous array of type `float32` that stores the values, the feature matrix
 *                  provides access to
 * @param numRows   The number of rows in the feature matrix
 * @param numCols   The number of columns in the feature matrix
 * @return          An unique pointer to an object of type `ICContiguousFeatureMatrix` that has been created
 */
MLRLCOMMON_API std::unique_ptr<ICContiguousFeatureMatrix> createCContiguousFeatureMatrix(const float32* array,
                                                                                         uint32 numRows,
                                                                                         uint32 numCols);
