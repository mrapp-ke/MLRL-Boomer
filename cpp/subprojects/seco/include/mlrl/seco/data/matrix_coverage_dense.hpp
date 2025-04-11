/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/matrix_c_contiguous.hpp"
#include "mlrl/common/data/matrix_dense.hpp"
#include "mlrl/common/indices/index_vector_complete.hpp"
#include "mlrl/common/indices/index_vector_partial.hpp"

namespace seco {

    /**
     * A two-dimensional matrix that stores how often individual examples and labels have been covered in a C-contiguous
     * array.
     */
    class DenseCoverageMatrix final : public DenseMatrixDecorator<AllocatedCContiguousView<uint32>> {
        private:

            float64 sumOfUncoveredWeights_;

        public:

            /**
             * @param numRows               The number of rows in the matrix
             * @param numCols               The number of columns in the matrix
             * @param sumOfUncoveredWeights The sum of the weights of all examples and labels that have not been covered
             *                              yet
             */
            DenseCoverageMatrix(uint32 numRows, uint32 numCols, float64 sumOfUncoveredWeights);

            /**
             * Returns the sum of the weights of all examples and labels that have not been covered yet.
             *
             * @return The sum of the weights of all examples and labels that have not been covered yet
             */
            float64 getSumOfUncoveredWeights() const;

            /**
             * Increases the number of times the elements at a specific row of this matrix are covered, given the
             * predictions of a rule that predicts for all available labels.
             *
             * @param row                        The row
             * @param majorityLabelIndicesBegin  An iterator to the beginning of the indices of the labels that are
             *                                   relevant to the majority of the training examples
             * @param majorityLabelIndicesEnd    An iterator to the end of the indices of the labels that are relevant
             *                                   to the majority of the training examples
             * @param predictionBegin            An iterator to the beginning of the predictions
             * @param predictionEnd              An iterator to the end of the predictions
             * @param indicesBegin               An iterator to the beginning of the label indices
             * @param indicesEnd                 An iterator to the end of the label indices
             */
            void increaseCoverage(uint32 row, View<uint32>::const_iterator majorityLabelIndicesBegin,
                                  View<uint32>::const_iterator majorityLabelIndicesEnd,
                                  View<uint8>::const_iterator predictionBegin,
                                  View<uint8>::const_iterator predictionEnd,
                                  CompleteIndexVector::const_iterator indicesBegin,
                                  CompleteIndexVector::const_iterator indicesEnd);

            /**
             * Increases the number of times the elements at a specific row of this matrix are covered, given the
             * predictions of a rule that predicts for a subset of the available labels.
             *
             * @param row                        The row
             * @param majorityLabelIndicesBegin  An iterator to the beginning of the indices of the labels that are
             *                                   relevant to the majority of the training examples
             * @param majorityLabelIndicesEnd    An iterator to the end of the indices of the labels that are relevant
             *                                   to the majority of the training examples
             * @param predictionBegin            An iterator to the beginning of the predictions
             * @param predictionEnd              An iterator to the end of the predictions
             * @param indicesBegin               An iterator to the beginning of the label indices
             * @param indicesEnd                 An iterator to the end of the label indices
             */
            void increaseCoverage(uint32 row, View<uint32>::const_iterator majorityLabelIndicesBegin,
                                  View<uint32>::const_iterator majorityLabelIndicesEnd,
                                  View<uint8>::const_iterator predictionBegin,
                                  View<uint8>::const_iterator predictionEnd,
                                  PartialIndexVector::const_iterator indicesBegin,
                                  PartialIndexVector::const_iterator indicesEnd);

            /**
             * Decreases the number of times the elements at a specific row of this matrix are covered, given the
             * predictions of a rule that predicts for all available labels.
             *
             * @param row                        The row
             * @param majorityLabelIndicesBegin  An iterator to the beginning of the indices of the labels that are
             *                                   relevant to the majority of the training examples
             * @param majorityLabelIndicesEnd    An iterator to the end of the indices of the labels that are relevant
             *                                   to the majority of the training examples
             * @param predictionBegin            An iterator to the beginning of the predictions
             * @param predictionEnd              An iterator to the end of the predictions
             * @param indicesBegin               An iterator to the beginning of the label indices
             * @param indicesEnd                 An iterator to the end of the label indices
             */
            void decreaseCoverage(uint32 row, View<uint32>::const_iterator majorityLabelIndicesBegin,
                                  View<uint32>::const_iterator majorityLabelIndicesEnd,
                                  View<uint8>::const_iterator predictionBegin,
                                  View<uint8>::const_iterator predictionEnd,
                                  CompleteIndexVector::const_iterator indicesBegin,
                                  CompleteIndexVector::const_iterator indicesEnd);

            /**
             * Decreases the number of times the elements at a specific row of this matrix are covered, given the
             * predictions of a rule that predicts for a subset of the available labels.
             *
             * @param row                        The row
             * @param majorityLabelIndicesBegin  An iterator to the beginning of the indices of the labels that are
             *                                   relevant to the majority of the training examples
             * @param majorityLabelIndicesEnd    An iterator to the end of the indices of the labels that are relevant
             *                                   to the majority of the training examples
             * @param predictionBegin            An iterator to the beginning of the predictions
             * @param predictionEnd              An iterator to the end of the predictions
             * @param indicesBegin               An iterator to the beginning of the label indices
             * @param indicesEnd                 An iterator to the end of the label indices
             */
            void decreaseCoverage(uint32 row, View<uint32>::const_iterator majorityLabelIndicesBegin,
                                  View<uint32>::const_iterator majorityLabelIndicesEnd,
                                  View<uint8>::const_iterator predictionBegin,
                                  View<uint8>::const_iterator predictionEnd,
                                  PartialIndexVector::const_iterator indicesBegin,
                                  PartialIndexVector::const_iterator indicesEnd);
    };

}
