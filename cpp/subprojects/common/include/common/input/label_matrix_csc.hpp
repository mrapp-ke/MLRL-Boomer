/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/view_csc_binary.hpp"
#include "common/input/label_matrix_c_contiguous.hpp"
#include "common/input/label_matrix_csr.hpp"


/**
 * Implements column-wise read-only access to the labels of individual training examples that are stored in a matrix in
 * the compressed sparse column (CSC) format.
 */
class CscLabelMatrix final {

    private:

        uint32* rowIndices_;

        uint32* colIndices_;

        BinaryCscView view_;

    public:

        /**
         * @param labelMatrix A reference to an object of type `CContiguousLabelMatrix` to be copied
         */
        CscLabelMatrix(const CContiguousLabelMatrix& labelMatrix);

        /**
         * @param labelMatrix A reference to an object of type `CsrLabelMatrix` to be copied
         */
        CscLabelMatrix(const CsrLabelMatrix& labelMatrix);

        ~CscLabelMatrix();

        /**
         * An iterator that provides access to the indices of the relevant labels and allows to modify them.
         */
        typedef BinaryCscView::index_iterator index_iterator;

        /**
         * An iterator that provides read-only access to the indices of the relevant labels.
         */
        typedef BinaryCscView::index_const_iterator index_const_iterator;

        /**
         * An iterator that provides read-only access to the values in the label matrix.
         */
        typedef BinaryCscView::value_const_iterator value_const_iterator;

        /**
         * Returns an `index_iterator` to the beginning of the indices at a specific column.
         *
         * @param col   The column
         * @return      An `index_iterator` to the beginning of the indices
         */
        index_iterator column_indices_begin(uint32 col);

        /**
         * Returns an `index_iterator` to the end of the indices at a specific column.
         *
         * @param col   The column
         * @return      An `index_iterator` to the end of the indices
         */
        index_iterator column_indices_end(uint32 col);

        /**
         * Returns an `index_const_iterator` to the beginning of the indices at a specific column.
         *
         * @param col   The column
         * @return      An `index_const_iterator` to the beginning of the indices
         */
        index_const_iterator column_indices_cbegin(uint32 col) const;

        /**
         * Returns an `index_const_iterator` to the end of the indices at a specific column.
         *
         * @param col   The column
         * @return      An `index_const_iterator` to the end of the indices
         */
        index_const_iterator column_indices_cend(uint32 col) const;

        /**
         * Returns a `value_const_iterator` to the beginning of the values at a specific column.
         *
         * @param row   The row
         * @return      A `value_const_iterator` to the beginning of the values
         */
        value_const_iterator column_values_cbegin(uint32 col) const;

        /**
         * Returns a `value_const_iterator` to the end of the values at a specific column.
         *
         * @param row   The row
         * @return      A `value_const_iterator` to the end of the values
         */
        value_const_iterator column_values_cend(uint32 col) const;

        /**
         * Returns the number of rows in the label matrix.
         *
         * @return The number of rows
         */
        uint32 getNumRows() const;

        /**
         * Returns the number of columns in the label matrix.
         *
         * @return The number of columns
         */
        uint32 getNumCols() const;

        /**
         * Returns the number of relevant labels.
         *
         * @return The number of relevant labels
         */
        uint32 getNumNonZeroElements() const;

};
