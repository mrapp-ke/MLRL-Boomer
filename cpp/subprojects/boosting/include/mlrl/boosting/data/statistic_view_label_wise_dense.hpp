/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/matrix_c_contiguous.hpp"
#include "mlrl/common/data/tuple.hpp"

namespace boosting {

    /**
     * Implements row-wise read and write access to the gradients and Hessians that have been calculated using a
     * label-wise decomposable loss function and are stored in pre-allocated C-contiguous arrays.
     */
    class DenseLabelWiseStatisticView : public CContiguousView<Tuple<float64>> {
        public:

            /**
             * @param numRows       The number of rows in the view
             * @param numCols       The number of columns in the view
             * @param statistics    A pointer to a C-contiguous array fo type `Tuple<float64>` that stores the gradients
             *                      and Hessians, the view provides access to
             */
            DenseLabelWiseStatisticView(uint32 numRows, uint32 numCols, Tuple<float64>* statistics);

            virtual ~DenseLabelWiseStatisticView() override {}

            /**
             * Sets all gradients and Hessians in the matrix to zero.
             */
            void clear();

            /**
             * Adds all gradients and Hessians in a vector to a specific row of this matrix. The gradients and Hessians
             * to be added are multiplied by a specific weight.
             *
             * @param row       The row
             * @param begin     An iterator to the beginning of the vector
             * @param end       An iterator to the end of the vector
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void addToRow(uint32 row, value_const_iterator begin, value_const_iterator end, float64 weight);

            /**
             * Returns the number of rows in the view.
             *
             * @return The number of rows
             */
            uint32 getNumRows() const;

            /**
             * Returns the number of columns in the view.
             *
             * @return The number of columns
             */
            uint32 getNumCols() const;
    };

}
