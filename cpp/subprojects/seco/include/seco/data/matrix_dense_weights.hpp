/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/matrix_dense.hpp"


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

    };

}
