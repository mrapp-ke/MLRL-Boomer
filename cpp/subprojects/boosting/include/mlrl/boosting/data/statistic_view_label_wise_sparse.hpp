/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/matrix_sparse_set.hpp"
#include "mlrl/common/data/tuple.hpp"

namespace boosting {

    /**
     * Implements row-wise read and write access to the gradients and Hessians that have been calculated using a
     * label-wise decomposable loss function and are stored in a pre-allocated matrix in the list of lists (LIL) format.
     */
    class SparseLabelWiseStatisticView : public SparseSetView<Tuple<float64>> {
        public:

            /**
             * @param numCols       The number of columns in the view
             * @param statistics    A pointer to an object of type `SparseSetMatrix` that stores the gradients and
             *                      Hessians
             */
            SparseLabelWiseStatisticView(ListOfLists<IndexedValue<Tuple<float64>>>&& valueView,
                                         CContiguousView<uint32>&& indexView, uint32 numRows, uint32 numCols);

            virtual ~SparseLabelWiseStatisticView() override {}

            /**
             * Sets all gradients and Hessians in the matrix to zero.
             */
            void clear();

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
