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
             * The type of a row in the view.
             */
            typedef LilMatrix<Tuple<float64>>::Row Row;

            /**
             * An iterator that provides read-only access to the elements in the view.
             */
            typedef Row::const_iterator const_iterator;

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

            /**
             * Returns a const reference to a specific row in the view.
             *
             * @param row   The row
             * @return      A const reference to the row
             */
            const Row& getRow(uint32 row) const;

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

            /**
             * Returns a reference to a specific row in the view.
             *
             * @param row   The row
             * @return      A reference to the row
             */
            Row& getRow(uint32 row);

            /**
             * Sets all gradients and Hessians in the matrix to zero.
             */
            void clear();

            /**
             * Adds all gradients and Hessians in a vector to a specific row of this matrix. The gradients and Hessians
             * to be added are multiplied by a specific weight.
             *
             * @param row       The row
             * @param begin     A `const_iterator` to the beginning of the vector
             * @param end       A `const_iterator` to the end of the vector
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void addToRow(uint32 row, const_iterator begin, const_iterator end, float64 weight);

    };

}
