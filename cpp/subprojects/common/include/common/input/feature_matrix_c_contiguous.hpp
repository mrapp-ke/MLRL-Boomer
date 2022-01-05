/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/input/feature_matrix_row_wise.hpp"
#include "common/data/view_c_contiguous.hpp"


/**
 * Defines an interface for all feature matrices that provide row-wise access to the feature values of individual
 * examples that are stored in a C-contiguous array.
 */
class ICContiguousFeatureMatrix : public IRowWiseFeatureMatrix {

    public:

        virtual ~ICContiguousFeatureMatrix() override { };

};

/**
 * An implementation of the type `ICContiguousFeatureMatrix` that provides row-wise read-only access to the feature
 * values of individual examples that are stored in a C-contiguous array.
 */
class CContiguousFeatureMatrix final : public ICContiguousFeatureMatrix {

    private:

        CContiguousConstView<const float32> view_;

    public:

        /**
         * @param numRows   The number of rows in the feature matrix
         * @param numCols   The number of columns in the feature matrix
         * @param array     A pointer to a C-contiguous array of type `float32` that stores the values, the feature
         *                  matrix provides access to
         */
        CContiguousFeatureMatrix(uint32 numRows, uint32 numCols, const float32* array);

        /**
         * An iterator that provides read-only access to the values in the feature matrix.
         */
        typedef const float32* const_iterator;

        /**
         * Returns a `const_iterator` to the beginning of a specific row.
         *
         * @param row   The row
         * @return      A `const_iterator` to the beginning of the given row
         */
        const_iterator row_cbegin(uint32 row) const;

        /**
         * Returns a `const_iterator` to the end of a specific row.
         *
         * @param row   The row
         * @return      A `const_iterator` to the end of the given row
         */
        const_iterator row_cend(uint32 row) const;

        uint32 getNumRows() const override;

        uint32 getNumCols() const override;

        bool isSparse() const override;

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
 * Creates and returns a new object of the type `ICContiguousFeatureMatrix`.
 *
 * @param numRows   The number of rows in the feature matrix
 * @param numCols   The number of columns in the feature matrix
 * @param array     A pointer to a C-contiguous array of type `float32` that stores the values, the feature matrix
 *                  provides access to
 * @return          An unique pointer to an object of type `ICContiguousFeatureMatrix` that has been created
 */
std::unique_ptr<ICContiguousFeatureMatrix> createCContiguousFeatureMatrix(uint32 numRows, uint32 numCols,
                                                                          const float32* array);
