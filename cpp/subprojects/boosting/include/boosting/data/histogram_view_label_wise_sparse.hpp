/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/triple.hpp"
#include "common/data/vector_dense.hpp"
#include "boosting/data/statistic_view_label_wise_sparse.hpp"


namespace boosting {

    /**
     * Implements row-wise read-only access to the gradients and Hessians that have been calculated using a label-wise
     * decomposable loss function and are stored in a pre-allocated histogram in the list of lists (LIL) format.
     */
    class SparseLabelWiseHistogramConstView {

        protected:

            /**
             * The number of columns in the view.
             */
            uint32 numCols_;

            /**
             * A pointer to an object of type `LilMatrix` that stores the gradients and Hessians.
             */
            LilMatrix<Triple<float64>>* histogram_;

            /**
             * A pointer to an object of type `DenseVector` that stores the weight of each bin.
             */
            DenseVector<float64>* weights_;

        public:

            /**
             * @param numCols   The number of columns in the view
             * @param histogram A pointer to an object of type `LilMatrix` that stores the gradients and Hessians
             * @param weights   A pointer to an object of type `DenseVector` that stores the weight of each bin
             */
            SparseLabelWiseHistogramConstView(uint32 numCols, LilMatrix<Triple<float64>>* histogram,
                                              DenseVector<float64>* weights);

            /**
             * The type of a row.
             */
            typedef LilMatrix<Triple<float64>>::Row Row;

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
             * Returns the weight of a specific row.
             *
             * @param row   The row
             * @return      The weight of the row
             */
            const float64 getWeight(uint32 row) const;

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
     * label-wise decomposable loss function and are stored in a pre-allocated histogram in the list of lists (LIL)
     * format.
     */
    class SparseLabelWiseHistogramView : public SparseLabelWiseHistogramConstView {

        public:

            /**
             * @param numCols   The number of columns in the view
             * @param histogram A pointer to an object of type `LilMatrix` that stores the gradients and Hessians
             * @param weights   A pointer to an object of type `DenseVector` that stores the weight of each bin
             */
            SparseLabelWiseHistogramView(uint32 numCols, LilMatrix<Triple<float64>>* histogram,
                                         DenseVector<float64>* weights);

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

            /**
             * Adds all gradients and Hessians in a vector to a specific row of this histogram. The gradients and
             * Hessians to be added are multiplied by a specific weight.
             *
             * @param row       The row
             * @param begin     A `SparseLabelWiseStatisticConstView::const_iterator` to the beginning of the vector
             * @param end       A `SparseLabelWiseStatisticConstView::const_iterator` to the end of the vector
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void addToRow(uint32 row, SparseLabelWiseStatisticConstView::const_iterator begin,
                          SparseLabelWiseStatisticConstView::const_iterator end, float64 weight);

    };

}
