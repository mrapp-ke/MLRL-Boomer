/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/view_csr_binary.hpp"
#include "common/input/label_matrix.hpp"


/**
 * Implements row-wise read-only access to the labels of individual training examples that are stored in a pre-allocated
 * sparse matrix in the compressed sparse row (CSR) format.
 */
class CsrLabelMatrix final : public ILabelMatrix {

    private:

        BinaryCsrView view_;

    public:

        /**
         * @param numRows       The number of rows in the label matrix
         * @param numCols       The number of columns in the label matrix
         * @param rowIndices    A pointer to an array of type `uint32`, shape `(numRows + 1)`, that stores the indices
         *                      of the first element in `colIndices` that corresponds to a certain row. The index at the
         *                      last position is equal to `num_non_zero_values`
         * @param colIndices    A pointer to an array of type `uint32`, shape `(num_non_zero_values)`, that stores the
         *                      column-indices, the relevant labels correspond to
         */
        CsrLabelMatrix(uint32 numRows, uint32 numCols, uint32* rowIndices, uint32* colIndices);

        /**
         * An iterator that provides read-only access to the indices of the relevant labels.
         */
        typedef BinaryCsrView::index_const_iterator index_const_iterator;

        /**
         * An iterator that provides read-only access to the values in the label matrix.
         */
        typedef BinaryCsrView::value_const_iterator value_const_iterator;

        /**
         * Returns an `index_const_iterator` to the beginning of the indices at a specific row.
         *
         * @param row   The row
         * @return      An `index_const_iterator` to the beginning of the indices
         */
        index_const_iterator row_indices_cbegin(uint32 row) const;

        /**
         * Returns an `index_const_iterator` to the end of the indices at a specific row.
         *
         * @param row   The row
         * @return      An `index_const_iterator` to the end of the indices
         */
        index_const_iterator row_indices_cend(uint32 row) const;

        /**
         * Returns a `value_const_iterator` to the beginning of the values at a specific row.
         *
         * @param row   The row
         * @return      A `value_const_iterator` to the beginning of the values
         */
        value_const_iterator row_values_cbegin(uint32 row) const;

        /**
         * Returns a `value_const_iterator` to the end of the values at a specific row.
         *
         * @param row   The row
         * @return      A `value_const_iterator` to the end of the values
         */
        value_const_iterator row_values_cend(uint32 row) const;

        /**
         * Returns the number of relevant labels.
         *
         * @return The number of relevant labels
         */
        uint32 getNumNonZeroElements() const;

        uint32 getNumRows() const override;

        uint32 getNumCols() const override;

        std::unique_ptr<LabelVector> getLabelVector(uint32 row) const override;

        std::unique_ptr<IStatisticsProvider> createStatisticsProvider(
            const IStatisticsProviderFactory& factory) const override;

};
