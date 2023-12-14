/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/iterator/diagonal_iterator.hpp"
#include "mlrl/common/data/view_matrix_c_contiguous.hpp"
#include "mlrl/common/data/view_matrix_composite.hpp"

namespace boosting {

    /**
     * Implements row-wise read and write access to the gradients and Hessians that have been calculated using a
     * non-decomposable loss function and are stored in pre-allocated C-contiguous arrays.
     */
    class DenseExampleWiseStatisticView : public CompositeMatrix<CContiguousView<float64>, CContiguousView<float64>> {
        public:

            /**
             * @param firstView     A reference to an object of type `CContiguousView` that stores the gradients
             * @param secondView    A reference to an object of type `CContiguousView` that stores the Hessians
             * @param numRows       The number of rows in the view
             * @param numCols       The number of columns in the view
             */
            DenseExampleWiseStatisticView(CContiguousView<float64>&& firstView, CContiguousView<float64>&& secondView,
                                          uint32 numRows, uint32 numCols);

            virtual ~DenseExampleWiseStatisticView() override {}

            /**
             * An iterator that provides read-only access to the gradients.
             */
            typedef CContiguousView<float64>::value_const_iterator gradient_const_iterator;

            /**
             * An iterator that provides access to the gradients and allows to modify them.
             */
            typedef CContiguousView<float64>::value_iterator gradient_iterator;

            /**
             * An iterator that provides read-only access to the Hessians.
             */
            typedef CContiguousView<float64>::value_const_iterator hessian_const_iterator;

            /**
             * An iterator that provides access to the Hessians and allows to modify them.
             */
            typedef CContiguousView<float64>::value_iterator hessian_iterator;

            /**
             * An iterator that provides read-only access to the Hessians that correspond to the diagonal of the matrix.
             */
            typedef DiagonalConstIterator<float64> hessian_diagonal_const_iterator;

            /**
             * Returns a `gradient_const_iterator` to the beginning of the gradients at a specific row.
             *
             * @param row   The row
             * @return      A `gradient_const_iterator` to the beginning of the given row
             */
            gradient_const_iterator gradients_cbegin(uint32 row) const;

            /**
             * Returns a `gradient_const_iterator` to the end of the gradients at a specific row.
             *
             * @param row   The row
             * @return      A `gradient_const_iterator` to the end of the given row
             */
            gradient_const_iterator gradients_cend(uint32 row) const;

            /**
             * Returns a `gradient_iterator` to the beginning of the gradients at a specific row.
             *
             * @param row   The row
             * @return      A `gradient_iterator` to the beginning of the given row
             */
            gradient_iterator gradients_begin(uint32 row);

            /**
             * Returns a `gradient_iterator` to the end of the gradients at a specific row.
             *
             * @param row   The row
             * @return      A `gradient_iterator` to the end of the given row
             */
            gradient_iterator gradients_end(uint32 row);

            /**
             * Returns a `hessian_const_iterator` to the beginning of the Hessians at a specific row.
             *
             * @param row   The row
             * @return      A `hessian_const_iterator` to the beginning of the given row
             */
            hessian_const_iterator hessians_cbegin(uint32 row) const;

            /**
             * Returns a `hessian_const_iterator` to the end of the Hessians at a specific row.
             *
             * @param row   The row
             * @return      A `hessian_const_iterator` to the end of the given row
             */
            hessian_const_iterator hessians_cend(uint32 row) const;

            /**
             * Returns a `hessian_diagonal_const_iterator` to the beginning of the Hessians that correspond to the
             * diagonal of the Hessian matrix at a specific row.
             *
             * @param row   The row
             * @return      A `hessian_diagonal_const_iterator` to the beginning
             */
            hessian_diagonal_const_iterator hessians_diagonal_cbegin(uint32 row) const;

            /**
             * Returns a `hessian_diagonal_const_iterator` to the end of the Hessians that correspond to the diagonal of
             * the Hessian matrix at a specific row.
             *
             * @param row   The row
             * @return      A `hessian_diagonal_const_iterator` to the end
             */
            hessian_diagonal_const_iterator hessians_diagonal_cend(uint32 row) const;

            /**
             * Returns a `hessian_iterator` to the beginning of the Hessians at a specific row.
             *
             * @param row   The row
             * @return      A `hessian_iterator` to the beginning of the given row
             */
            hessian_iterator hessians_begin(uint32 row);

            /**
             * Returns a `hessian_iterator` to the end of the Hessians at a specific row.
             *
             * @param row   The row
             * @return      A `hessian_iterator` to the end of the given row
             */
            hessian_iterator hessians_end(uint32 row);

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
