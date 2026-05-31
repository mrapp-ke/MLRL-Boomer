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
        : public CompositeMatrix<CompositeMatrix<AllocatedListOfLists<uint32>, AllocatedListOfLists<uint32>>,
                                 CompositeMatrix<AllocatedListOfLists<uint32>, AllocatedListOfLists<uint32>>> {
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
             * Returns an `index_const_iterator` to the beginning of the indices at a specific row that corresponds to
             * irrelevant labels for which a rule predicts negatively.
             *
             * @param row   The row
             * @return      An `index_const_iterator` to the beginning of the given row
             */
            index_const_iterator in_indices_cbegin(uint32 row) const;

            /**
             * Returns an `index_const_iterator` to the end of the indices at a specific row that correspond to
             * irrelevant labels for which a rule predicts negatively.
             *
             * @param row   The row
             * @return      An `index_const_iterator` to the end of the given row
             */
            index_const_iterator in_indices_cend(uint32 row) const;

            /**
             * Returns an `index_iterator` to the beginning of the indices at a specific row that corresponds to
             * irrelevant labels for which a rule predicts negatively.
             *
             * @param row   The row
             * @return      An `index_iterator` to the beginning of the given row
             */
            index_iterator in_indices_begin(uint32 row);

            /**
             * Returns an `index_iterator` to the end of the indices at a specific row that correspond to irrelevant
             * labels for which a rule predicts negatively.
             *
             * @param row   The row
             * @return      An `index_iterator` to the end of the given row
             */
            index_iterator in_indices_end(uint32 row);

            /**
             * Returns a view that provides read-only access to a specific row in the view that corresponds to
             * irrelevant labels for which a rule predicts negatively.
             *
             * @param row   The index of the row
             * @return      A `const_row`
             */
            const_row in_const_row(uint32 row) const;

            /**
             * Returns a view that provides access to a specific row in the view that corresponds to irrelevant labels
             * for which a rule predicts negatively and allows to modify it.
             *
             * @param row   The index of the row
             * @return      A `row`
             */
            row in_row(uint32 row);

            /**
             * Returns an `index_const_iterator` to the beginning of the indices at a specific row that corresponds to
             * irrelevant labels for which a rule predicts positively.
             *
             * @param row   The row
             * @return      An `index_const_iterator` to the beginning of the given row
             */
            index_const_iterator ip_indices_cbegin(uint32 row) const;

            /**
             * Returns an `index_const_iterator` to the end of the indices at a specific row that correspond to
             * irrelevant labels for which a rule predicts positively.
             *
             * @param row   The row
             * @return      An `index_const_iterator` to the end of the given row
             */
            index_const_iterator ip_indices_cend(uint32 row) const;

            /**
             * Returns an `index_iterator` to the beginning of the indices at a specific row that corresponds to
             * irrelevant labels for which a rule predicts positively.
             *
             * @param row   The row
             * @return      An `index_iterator` to the beginning of the given row
             */
            index_iterator ip_indices_begin(uint32 row);

            /**
             * Returns an `index_iterator` to the end of the indices at a specific row that correspond to irrelevant
             * labels for which a rule predicts positively.
             *
             * @param row   The row
             * @return      An `index_iterator` to the end of the given row
             */
            index_iterator ip_indices_end(uint32 row);

            /**
             * Returns a view that provides read-only access to a specific row in the view that corresponds to
             * irrelevant labels for which a rule predicts positively.
             *
             * @param row   The index of the row
             * @return      A `const_row`
             */
            const_row ip_const_row(uint32 row) const;

            /**
             * Returns a view that provides access to a specific row in the view that corresponds to irrelevant labels
             * for which a rule predicts positively and allows to modify it.
             *
             * @param row   The index of the row
             * @return      A `row`
             */
            row ip_row(uint32 row);

            /**
             * Returns an `index_const_iterator` to the beginning of the indices at a specific row that corresponds to
             * relevant labels for which a rule predicts negatively.
             *
             * @param row   The row
             * @return      An `index_const_iterator` to the beginning of the given row
             */
            index_const_iterator rn_indices_cbegin(uint32 row) const;

            /**
             * Returns an `index_const_iterator` to the end of the indices at a specific row that correspond to relevant
             * labels for which a rule predicts negatively.
             *
             * @param row   The row
             * @return      An `index_const_iterator` to the end of the given row
             */
            index_const_iterator rn_indices_cend(uint32 row) const;

            /**
             * Returns an `index_iterator` to the beginning of the indices at a specific row that corresponds to
             * relevant labels for which a rule predicts negatively.
             *
             * @param row   The row
             * @return      An `index_iterator` to the beginning of the given row
             */
            index_iterator rn_indices_begin(uint32 row);

            /**
             * Returns an `index_iterator` to the end of the indices at a specific row that correspond to relevant
             * labels for which a rule predicts negatively.
             *
             * @param row   The row
             * @return      An `index_iterator` to the end of the given row
             */
            index_iterator rn_indices_end(uint32 row);

            /**
             * Returns a view that provides read-only access to a specific row in the view that corresponds to relevant
             * labels for which a rule predicts negatively.
             *
             * @param row   The index of the row
             * @return      A `const_row`
             */
            const_row rn_const_row(uint32 row) const;

            /**
             * Returns a view that provides access to a specific row in the view that corresponds to relevant labels for
             * which a rule predicts negatively and allows to modify it.
             *
             * @param row   The index of the row
             * @return      A `row`
             */
            row rn_row(uint32 row);

            /**
             * Returns an `index_const_iterator` to the beginning of the indices at a specific row that corresponds to
             * relevant labels for which a rule predicts positively.
             *
             * @param row   The row
             * @return      An `index_const_iterator` to the beginning of the given row
             */
            index_const_iterator rp_indices_cbegin(uint32 row) const;

            /**
             * Returns an `index_const_iterator` to the end of the indices at a specific row that correspond to relevant
             * labels for which a rule predicts positively.
             *
             * @param row   The row
             * @return      An `index_const_iterator` to the end of the given row
             */
            index_const_iterator rp_indices_cend(uint32 row) const;

            /**
             * Returns an `index_iterator` to the beginning of the indices at a specific row that corresponds to
             * relevant labels for which a rule predicts positively.
             *
             * @param row   The row
             * @return      An `index_iterator` to the beginning of the given row
             */
            index_iterator rp_indices_begin(uint32 row);

            /**
             * Returns an `index_iterator` to the end of the indices at a specific row that correspond to relevant
             * labels for which a rule predicts positively.
             *
             * @param row   The row
             * @return      An `index_iterator` to the end of the given row
             */
            index_iterator rp_indices_end(uint32 row);

            /**
             * Returns a view that provides read-only access to a specific row in the view that corresponds to relevant
             * labels for which a rule predicts positively.
             *
             * @param row   The index of the row
             * @return      A `const_row`
             */
            const_row rp_const_row(uint32 row) const;

            /**
             * Returns a view that provides access to a specific row in the view that corresponds to relevant labels for
             * which a rule predicts positively and allows to modify it.
             *
             * @param row   The index of the row
             * @return      A `row`
             */
            row rp_row(uint32 row);
    };
}
