/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/tuple.hpp"
#include "common/data/matrix_sparse_set.hpp"


namespace boosting {

    /**
     * Implements row-wise read-only access to the gradients and Hessians that have been calculated using a label-wise
     * decomposable loss function and are stored in a pre-allocated matrix in the list of lists (LIL) format.
     */
    class SparseLabelWiseStatisticConstView {

        protected:

            /**
             * The number of columns in the view.
             */
            uint32 numCols_;

            /**
             * A pointer to an object of type `SparseSetMatrix` that stores the gradients and Hessians.
             */
            SparseSetMatrix<Tuple<float64>>* statistics_;

        public:

            /**
             * @param numCols       The number of columns in the view
             * @param statistics    A pointer to an object of type `SparseSetMatrix` that stores the gradients and
             *                      Hessians
             */
            SparseLabelWiseStatisticConstView(uint32 numCols, SparseSetMatrix<Tuple<float64>>* statistics);

            /**
             * The type of a row.
             */
            typedef SparseSetMatrix<Tuple<float64>>::Row Row;

            /**
             * An iterator that provides read-only access to the elements in the view.
             */
            typedef Row::const_iterator const_iterator;

            /**
             * Returns a `const_iterator` to the beginning of a specific row.
             *
             * @param row   The row
             * @return      A `const_iterator` to the beginning
             */
            const_iterator row_cbegin(uint32 row) const;

            /**
             * Returns a `const_iterator` to the end of a specific row.
             *
             * @param row   The row
             * @return      A `const_iterator` to the end
             */
            const_iterator row_cend(uint32 row) const;

            /**
             * Returns a specific row.
             *
             * @param row   The index of the row to be returned
             * @return      The row
             */
            const Row getRow(uint32 row) const;

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
             * @param numCols       The number of columns in the view
             * @param statistics    A pointer to an object of type `SparseSetMatrix` that stores the gradients and
             *                      Hessians
             */
            SparseLabelWiseStatisticView(uint32 numCols, SparseSetMatrix<Tuple<float64>>* statistics);

            /**
             * Returns a specific row.
             *
             * @param row   The index of the row to be returned
             * @return      The row
             */
            Row getRow(uint32 row);

            /**
             * Sets all gradients and Hessians in the matrix to zero.
             */
            void clear();

    };

}
