/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_matrix_composite.hpp"
#include "mlrl/common/data/view_matrix_lil.hpp"
#include "mlrl/seco/util/dll_exports.hpp"

namespace seco {

    /**
     * Implements row-wise read and write access to confusion matrices that are stored in a pre-allocated sparse matrix
     * in the list of lists (LIL) format.
     */
    class MLRLSECO_API SparseDecomposableStatisticView final
        : public CompositeMatrix<AllocatedListOfLists<uint32>, AllocatedListOfLists<uint32>> {
        public:

            /**
             * @param numRows   The number of rows in the view
             * @param numCols   The number of columns in the view
             */
            SparseDecomposableStatisticView(uint32 numRows, uint32 numCols);

            /**
             * @param other A reference to an object of type `SparseDecomposableStatisticView` that should be copied
             */
            SparseDecomposableStatisticView(SparseDecomposableStatisticView&& other);

            virtual ~SparseDecomposableStatisticView() override {}

            /**
             * Provides read-only access to an individual row in the view.
             */
            using const_row = typename ListOfLists<uint32>::const_row;

            /**
             * Provides access to an individual row in the view and allows to modify it.
             */
            using row = typename ListOfLists<uint32>::row;

            /**
             * An iterator that provides access to the indices individual confusion matrix elements in the view
             * correspond to and allows to modify them.
             */
            using index_iterator = typename ListOfLists<uint32>::value_iterator;

            /**
             * An iterator that provides read-only access to the indices individual confusion matrix elements in the
             * view correspond to.
             */
            using index_const_iterator = typename ListOfLists<uint32>::value_const_iterator;

            /**
             * Returns an `index_const_iterator` to the beginning of the indices at a specific row that correspond to
             * the labels for which a rule predicts correctly.
             *
             * @param row   The row
             * @return      An `index_const_iterator` to the beginning of the given row
             */
            index_const_iterator correct_indices_cbegin(uint32 row) const;

            /**
             * Returns an `index_const_iterator` to the end of the indices at a specific row that correspond to the
             * labels for which a rule predicts correctly.
             *
             * @param row   The row
             * @return      An `index_const_iterator` to the end of the given row
             */
            index_const_iterator correct_indices_cend(uint32 row) const;

            /**
             * Returns an `index_iterator` to the beginning of the indices at a specific row that correspond to the
             * labels for which a rule predicts correctly.
             *
             * @param row   The row
             * @return      An `index_iterator` to the beginning of the given row
             */
            index_iterator correct_indices_begin(uint32 row);

            /**
             * Returns an `index_iterator` to the end of the indices at a specific row that correspond to the labels for
             * which a rule predicts correctly.
             *
             * @param row   The row
             * @return      An `index_iterator` to the end of the given row
             */
            index_iterator correct_indices_end(uint32 row);

            /**
             * Returns a view that provides read-only access to a specific row in the view that correspond to the labels
             * for which a rule predicts correctly.
             *
             * @param row   The index of the row
             * @return      A `const_row`
             */
            const_row correct_indices_const_row(uint32 row) const;

            /**
             * Returns a view that provides access to a specific row in the view that correspond to the labels for which
             * a rule predicts correctly and allows to modify it.
             *
             * @param row   The index of the row
             * @return      A `row`
             */
            row correct_indices_row(uint32 row);

            /**
             * Returns an `index_const_iterator` to the beginning of the indices at a specific row that correspond to
             * the labels for which a rule predicts incorrectly.
             *
             * @param row   The row
             * @return      An `index_const_iterator` to the beginning of the given row
             */
            index_const_iterator incorrect_indices_cbegin(uint32 row) const;

            /**
             * Returns an `index_const_iterator` to the end of the indices at a specific row that correspond to the
             * labels for which a rule predicts incorrectly.
             *
             * @param row   The row
             * @return      An `index_const_iterator` to the end of the given row
             */
            index_const_iterator incorrect_indices_cend(uint32 row) const;

            /**
             * Returns an `index_iterator` to the beginning of the indices at a specific row that correspond to the
             * labels for which a rule predicts incorrectly.
             *
             * @param row   The row
             * @return      An `index_iterator` to the beginning of the given row
             */
            index_iterator incorrect_indices_begin(uint32 row);

            /**
             * Returns an `index_iterator` to the end of the indices at a specific row that correspond to the labels for
             * which a rule predicts incorrectly.
             *
             * @param row   The row
             * @return      An `index_iterator` to the end of the given row
             */
            index_iterator incorrect_indices_end(uint32 row);

            /**
             * Returns a view that provides read-only access to a specific row in the view that correspond to the labels
             * for which a rule predicts incorrectly.
             *
             * @param row   The index of the row
             * @return      A `const_row`
             */
            const_row incorrect_indices_const_row(uint32 row) const;

            /**
             * Returns a view that provides access to a specific row in the view that correspond to the labels for which
             * a rule predicts incorrectly and allows to modify it.
             *
             * @param row   The index of the row
             * @return      A `row`
             */
            row incorrect_indices_row(uint32 row);
    };
}
