/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_matrix_bit.hpp"
#include "mlrl/common/data/view_matrix_composite.hpp"

namespace boosting {

    /**
     * Implements row-wise read and write access to the gradients and Hessians that have been calculated using a
     * decomposable loss function and are stored in pre-allocated bit matrices.
     */
    class BitDecomposableStatisticView
        : public CompositeMatrix<AllocatedBitMatrix<uint32>, AllocatedBitMatrix<uint32>> {
        public:

            /**
             * @param numRows   The number of rows in the view
             * @param numCols   The number of columns in the view
             * @param numBits   The number of bits per statistic in the view
             */
            BitDecomposableStatisticView(uint32 numRows, uint32 numCols, uint32 numBits);

            /**
             * @param other A reference to an object of type `BitDecomposableStatisticView` that should be copied
             */
            BitDecomposableStatisticView(BitDecomposableStatisticView&& other);

            virtual ~BitDecomposableStatisticView() override {}
    };

}
