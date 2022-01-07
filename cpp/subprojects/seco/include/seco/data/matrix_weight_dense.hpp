/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/matrix_dense.hpp"
#include "common/data/vector_sparse_array_binary.hpp"
#include "common/indices/index_vector_complete.hpp"
#include "common/indices/index_vector_partial.hpp"


namespace seco {

    /**
     * A two-dimensional matrix that stores the weights of individual examples and labels in a C-contiguous array.
     */
    class DenseWeightMatrix final : public DenseMatrix<float64> {

        private:

            float64 sumOfUncoveredWeights_;

        public:

            /**
             * @param numRows               The number of rows in the matrix
             * @param numCols               The number of columns in the matrix
             * @param sumOfUncoveredWeights The sum of the weights of all labels that remain to be covered
             */
            DenseWeightMatrix(uint32 numRows, uint32 numCols, float64 sumOfUncoveredWeights);

            /**
             * Returns the sum of the weights of all labels that remain to be covered.
             *
             * @return The sum of the weights
             */
            float64 getSumOfUncoveredWeights() const;

            /**
             * Updates the weights at a specific row of this matrix, given the predictions for certain labels.
             *
             * @param row                   The row
             * @param majorityLabelVector   A reference to an object of type `BinarySparseArrayVector` that stores the
             *                              prediction of the default rule
             * @param predictionBegin       An iterator to the beginning of the predictions
             * @param predictionEnd         An iterator to the end of the predictions
             * @param indicesBegin          An iterator to the beginning of the label indices
             * @param indicesEnd            An iterator to the end of the label indices
             */
            void updateRow(uint32 row, const BinarySparseArrayVector& majorityLabelVector,
                           VectorView<float64>::const_iterator predictionBegin,
                           VectorView<float64>::const_iterator predictionEnd,
                           CompleteIndexVector::const_iterator indicesBegin,
                           CompleteIndexVector::const_iterator indicesEnd);

            /**
             * Updates the weights at a specific row of this matrix, given the predictions for certain labels.
             *
             * @param row                   The row
             * @param majorityLabelVector   A reference to an object of type `BinarySparseArrayVector` that stores the
             *                              prediction of the default rule
             * @param predictionBegin       An iterator to the beginning of the predictions
             * @param predictionEnd         An iterator to the end of the predictions
             * @param indicesBegin          An iterator to the beginning of the label indices
             * @param indicesEnd            An iterator to the end of the label indices
             */
            void updateRow(uint32 row, const BinarySparseArrayVector& majorityLabelVector,
                           VectorView<float64>::const_iterator predictionBegin,
                           VectorView<float64>::const_iterator predictionEnd,
                           PartialIndexVector::const_iterator indicesBegin,
                           PartialIndexVector::const_iterator indicesEnd);

    };

}
