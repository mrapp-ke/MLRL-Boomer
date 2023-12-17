/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#ifdef _WIN32
    #pragma warning(push)
    #pragma warning(disable : 4250)
#endif

#include "mlrl/common/input/feature_matrix_row_wise.hpp"

/**
 * Defines an interface for all feature matrices that provide row-wise access to the feature values of examples that are
 * stored in a sparse matrix in the compressed sparse row (CSR) format.
 */
class MLRLCOMMON_API ICsrFeatureMatrix : public IRowWiseFeatureMatrix {
    public:

        virtual ~ICsrFeatureMatrix() override {}
};

/**
 * Creates and returns a new object of the type `ICsrFeatureMatrix`.
 *
 * @param values        A pointer to an array of type `float32`, shape `(numNonZeroValues)`, that stores all non-zero
 *                      values
 * @param indices       A pointer to an array of type `uint32`, shape `(numNonZeroValues)`, that stores the
 *                      column-indices, the values in `values` correspond to
 * @param indptr        A pointer to an array of type `uint32`, shape `(numRows + 1)`, that stores the indices of the
 *                      first element in `values` and `indices` that corresponds to a certain row. The index at the last
 *                      position is equal to `numNonZeroValues`
 * @param numRows       The number of rows in the feature matrix
 * @param numCols       The number of columns in the feature matrix
 * @return              An unique pointer to an object of type `ICsrFeatureMatrix` that has been created
 */
MLRLCOMMON_API std::unique_ptr<ICsrFeatureMatrix> createCsrFeatureMatrix(const float32* values, uint32* indices,
                                                                         uint32* indptr, uint32 numRows,
                                                                         uint32 numCols);

#ifdef _WIN32
    #pragma warning(pop)
#endif
