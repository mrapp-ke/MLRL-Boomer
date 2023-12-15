/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/tuple.hpp"

namespace boosting {

    /**
     * Implements row-wise read and write access to the gradients and Hessians that have been calculated using a
     * label-wise decomposable loss function and are stored in pre-allocated C-contiguous arrays.
     */
    class DenseLabelWiseStatisticView {
        protected:

            /**
             * The number of rows in the view.
             */
            const uint32 numRows_;

            /**
             * The number of columns in the view.
             */
            const uint32 numCols_;

            /**
             * A pointer to an array that stores the gradients and Hessians.
             */
            Tuple<float64>* statistics_;

        public:

            /**
             * @param numRows       The number of rows in the view
             * @param numCols       The number of columns in the view
             * @param statistics    A pointer to a C-contiguous array fo type `Tuple<float64>` that stores the gradients
             *                      and Hessians, the view provides access to
             */
            DenseLabelWiseStatisticView(uint32 numRows, uint32 numCols, Tuple<float64>* statistics);

            virtual ~DenseLabelWiseStatisticView() {}

            /**
             * An iterator that provides read-only access to the elements in the view.
             */
            typedef const Tuple<float64>* value_const_iterator;

            /**
             * An iterator that provides access to the elements in the view and allows to modify them.
             */
            typedef Tuple<float64>* value_iterator;

            /**
             * Returns a `value_const_iterator` to the beginning of a specific row.
             *
             * @param row   The row
             * @return      A `value_const_iterator` to the beginning
             */
            value_const_iterator values_cbegin(uint32 row) const;

            /**
             * Returns a `value_const_iterator` to the end of a specific row.
             *
             * @param row   The row
             * @return      A `value_const_iterator` to the end
             */
            value_const_iterator values_cend(uint32 row) const;

            /**
             * Returns a `value_iterator` to the beginning of a specific row.
             *
             * @param row   The row
             * @return      A `value_iterator` to the beginning
             */
            value_iterator values_begin(uint32 row);

            /**
             * Returns a `value_iterator` to the end of a specific row.
             *
             * @param row   The row
             * @return      A `value_iterator` to the end
             */
            value_iterator values_end(uint32 row);

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
