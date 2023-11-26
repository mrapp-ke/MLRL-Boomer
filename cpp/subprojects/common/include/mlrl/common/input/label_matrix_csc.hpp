/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_matrix_c_contiguous.hpp"
#include "mlrl/common/data/view_matrix_csc_binary.hpp"
#include "mlrl/common/data/view_matrix_csr_binary.hpp"
#include "mlrl/common/indices/index_vector_complete.hpp"
#include "mlrl/common/indices/index_vector_partial.hpp"

/**
 * Implements column-wise read-only access to the labels of individual training examples that are stored in a matrix in
 * the compressed sparse column (CSC) format.
 *
 * This class provides copy constructors for copying an existing `CContiguousView`, which provides random access, or a
 * `BinaryCsrView`, which provides row-wise access to the labels of the training examples. These constructors expect the
 * indices of the examples to be considered when copying the existing label matrix to be provided.
 */
class CscLabelMatrix final : public BinaryCscView {
    public:

        /**
         * @param labelMatrix   A reference to an object of type `CContiguousView` to be copied
         * @param indicesBegin  A `CompleteIndexVector::const_iterator` to the beginning of the indices of the examples
         *                      to be considered
         * @param indicesEnd    A `CompleteIndexVector::const_iterator` to the end of the indices of the examples to be
         *                      considered
         */
        CscLabelMatrix(const CContiguousView<const uint8>& labelMatrix,
                       CompleteIndexVector::const_iterator indicesBegin,
                       CompleteIndexVector::const_iterator indicesEnd);

        /**
         * @param labelMatrix   A reference to an object of type `CContiguousView` to be copied
         * @param indicesBegin  A `PartialIndexVector::const_iterator` to the beginning of the indices of the examples
         *                      to be considered
         * @param indicesEnd    A `PartialIndexVector::const_iterator` to the end of the indices of the examples to be
         *                      considered
         */
        CscLabelMatrix(const CContiguousView<const uint8>& labelMatrix, PartialIndexVector::const_iterator indicesBegin,
                       PartialIndexVector::const_iterator indicesEnd);

        /**
         * @param labelMatrix   A reference to an object of type `BinaryCsrView` to be copied
         * @param indicesBegin  A `CompleteIndexVector::const_iterator` to the beginning of the indices of the examples
         *                      to be considered
         * @param indicesEnd    A `CompleteIndexVector::const_iterator` to the end of the indices of the examples to be
         *                      considered
         */
        CscLabelMatrix(const BinaryCsrView& labelMatrix, CompleteIndexVector::const_iterator indicesBegin,
                       CompleteIndexVector::const_iterator indicesEnd);

        /**
         * @param labelMatrix   A reference to an object of type `BinaryCsrView` to be copied
         * @param indicesBegin  A `PartialIndexVector::const_iterator` to the beginning of the indices of the examples
         *                      to be considered
         * @param indicesEnd    A `PartialIndexVector::const_iterator` to the end of the indices of the examples to be
         *                      considered
         */
        CscLabelMatrix(const BinaryCsrView& labelMatrix, PartialIndexVector::const_iterator indicesBegin,
                       PartialIndexVector::const_iterator indicesEnd);

        ~CscLabelMatrix() override;
};
