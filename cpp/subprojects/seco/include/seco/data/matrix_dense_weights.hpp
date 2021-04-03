/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/matrix_dense.hpp"
#include "common/data/vector_dense.hpp"
#include "common/indices/index_vector_full.hpp"
#include "common/indices/index_vector_partial.hpp"


namespace seco {

    /**
     * A two-dimensional matrix that stores the weights of individual examples and labels in a C-contiguous array.
     */
    class DenseWeightMatrix final : public DenseMatrix<uint8> {

        private:

            uint32 sumOfUncoveredWeights_;

        public:

            /**
             * @param numRows   The number of rows in the matrix
             * @param numCols   The number of columns in the matrix
             */
            DenseWeightMatrix(uint32 numRows, uint32 numCols);

            /**
             * Returns the sum of the weights of all labels that remain to be covered.
             *
             * @return The sum of the weights
             */
            uint32 getSumOfUncoveredWeights() const;

            /**
             * Sets the sum of the weights of all labels that remain to be covered.
             *
             * @param sumOfUncoveredWeights The sum of weights to be set
             */
            void setSumOfUncoveredWeights(uint32 sumOfUncoveredWeights);

            /**
             * Updates the weights at a specific row of this matrix, given the predictions for certain labels.
             *
             * @param row                   The row
             * @param majorityLabelVector   A reference to an object of type `DenseVector` that stores the prediction of
             *                              the default rule
             * @param begin                 An iterator to the beginning of the predictions
             * @param end                   An iterator to the end of the predictions
             * @param indicesBegin          An iterator to the beginning of the label indices
             * @param indicesEnd            An iterator to the end of the label indices
             */
            void updateRow(uint32 row, const DenseVector<uint8>& majorityLabelVector,
                           DenseVector<float64>::const_iterator predictionBegin,
                           DenseVector<float64>::const_iterator predictionEnd,
                           FullIndexVector::const_iterator indicesBegin,
                           FullIndexVector::const_iterator indicesEnd);

            /**
             * Updates the weights at a specific row of this matrix, given the predictions for certain labels.
             *
             * @param row                   The row
             * @param majorityLabelVector   A reference to an object of type `DenseVector` that stores the prediction of
             *                              the default rule
             * @param begin                 An iterator to the beginning of the predictions
             * @param end                   An iterator to the end of the predictions
             * @param indicesBegin          An iterator to the beginning of the label indices
             * @param indicesEnd            An iterator to the end of the label indices
             */
            void updateRow(uint32 row, const DenseVector<uint8>& majorityLabelVector,
                           DenseVector<float64>::const_iterator predictionBegin,
                           DenseVector<float64>::const_iterator predictionEnd,
                           PartialIndexVector::const_iterator indicesBegin,
                           PartialIndexVector::const_iterator indicesEnd);

    };

}
