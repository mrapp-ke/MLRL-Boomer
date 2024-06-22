/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/feature_matrix_column_wise.hpp"

#include <memory>

/**
 * Defines an interface for all feature matrices that provide column-wise access to the feature values of examples that
 * are stored in a Fortran-contiguous array.
 */
class MLRLCOMMON_API IFortranContiguousFeatureMatrix : public IColumnWiseFeatureMatrix {
    public:

        virtual ~IFortranContiguousFeatureMatrix() override {}
};

/**
 * Creates and returns a new object of type `IFortranContiguousFeatureMatrix`.
 *
 * @param array     A pointer to a Fortran-contiguous array of type `float32` that stores the feature values
 * @param numRows   The number of rows in the feature matrix
 * @param numCols   The number of columns in the feature matrix
 * @return          An unique pointer to an object of type `IFortranContiguousFeatureMatrix` that has been created
 */
MLRLCOMMON_API std::unique_ptr<IFortranContiguousFeatureMatrix> createFortranContiguousFeatureMatrix(
  const float32* array, uint32 numRows, uint32 numCols);
