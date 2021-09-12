/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/tuple.hpp"
#include "common/data/matrix_lil.hpp"


namespace boosting {

    /**
     * Implements row-wise read-only access to the gradients and Hessians that have been calculated using a label-wise
     * decomposable loss function and are stored in a pre-allocated matrix in the list of lists (LIL) format.
     */
    class SparseLabelWiseStatisticConstView {

        protected:

            /**
             * A pointer to an object of type `LilMatrix` that stores the gradients and Hessians.
             */
            LilMatrix<Tuple<float64>>* statistics_;

            /**
             * The number of columns in the view.
             */
            uint32 numCols_;

        public:

            /**
             * @param statistics    A pointer to an object of type `LilMatrix` that stores the gradients and Hessians
             * @param numCols       The number of columns in the view
             */
            SparseLabelWiseStatisticConstView(LilMatrix<Tuple<float64>>* statistics, uint32 numCols);

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

    /**
     * Implements row-wise read and write access to the gradients and Hessians that have been calculated using a
     * label-wise decomposable loss function and are stored in a pre-allocated matrix in the list of lists (LIL) format.
     */
    class SparseLabelWiseStatisticView : public SparseLabelWiseStatisticConstView {

        public:

            /**
             * @param statistics    A pointer to an object of type `LilMatrix` that stores the gradients and Hessians
             * @param numCols       The number of columns in the view
             */
            SparseLabelWiseStatisticView(LilMatrix<Tuple<float64>>* statistics, uint32 numCols);

    };

}
