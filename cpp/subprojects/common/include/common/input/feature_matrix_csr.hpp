/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/input/feature_matrix_row_wise.hpp"
#include "common/data/view_csr.hpp"


/**
 * Defines an interface for all feature matrices that provide row-wise access to the feature values of individual
 * examples that are stored in a sparse matrix in the compressed sparse row (CSR) format.
 */
class ICsrFeatureMatrix : virtual public IRowWiseFeatureMatrix {

    public:

        virtual ~ICsrFeatureMatrix() override { };

};

/**
 * An implementation of the type `ICsrFeatureMatrix` that provides row-wise read-only access to the feature values of
 * individual examples that are stored in a sparse matrix in the compressed sparse row (CSR) format.
 */
class CsrFeatureMatrix final : public CsrConstView<const float32>, virtual public ICsrFeatureMatrix {

    public:

        /**
         * @param numRows       The number of rows in the feature matrix
         * @param numCols       The number of columns in the feature matrix
         * @param data          A pointer to an array of type `float32`, shape `(num_non_zero_values)`, that stores all
         *                      non-zero values
         * @param rowIndices    A pointer to an array of type `uint32`, shape `(numRows + 1)`, that stores the indices
         *                      of the first element in `data` and `colIndices` that corresponds to a certain row. The
         *                      index at the last position is equal to `num_non_zero_values`
         * @param colIndices    A pointer to an array of type `uint32`, shape `(num_non_zero_values)`, that stores the
         *                      column-indices, the values in `data` correspond to
         */
        CsrFeatureMatrix(uint32 numRows, uint32 numCols, const float32* data, uint32* rowIndices, uint32* colIndices);

        std::unique_ptr<DensePredictionMatrix<uint8>> predictLabels(const IClassificationPredictor& predictor,
                                                                    uint32 numLabels) const override;

        std::unique_ptr<BinarySparsePredictionMatrix> predictSparseLabels(const IClassificationPredictor& predictor,
                                                                          uint32 numLabels) const override;

        std::unique_ptr<DensePredictionMatrix<float64>> predictScores(const IRegressionPredictor& predictor,
                                                                      uint32 numLabels) const override;

        std::unique_ptr<DensePredictionMatrix<float64>> predictProbabilities(const IProbabilityPredictor& predictor,
                                                                             uint32 numLabels) const override;

};

/**
 * Creates and returns a new object of the type `ICsrFeatureMatrix`.
 *
 * @param numRows       The number of rows in the feature matrix
 * @param numCols       The number of columns in the feature matrix
 * @param data          A pointer to an array of type `float32`, shape `(num_non_zero_values)`, that stores all non-zero
 *                      values
 * @param rowIndices    A pointer to an array of type `uint32`, shape `(numRows + 1)`, that stores the indices of the
 *                      first element in `data` and `colIndices` that corresponds to a certain row. The index at the
 *                      last position is equal to `num_non_zero_values`
 * @param colIndices    A pointer to an array of type `uint32`, shape `(num_non_zero_values)`, that stores the
 *                      column-indices, the values in `data` correspond to
 * @return              An unique pointer to an object of type `ICsrFeatureMatrix` that has been created
 */
std::unique_ptr<ICsrFeatureMatrix> createCsrFeatureMatrix(uint32 numRows, uint32 numCols, const float32* data,
                                                          uint32* rowIndices, uint32* colIndices);
