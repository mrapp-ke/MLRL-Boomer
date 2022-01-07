/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/view_csc_binary.hpp"
#include "common/input/label_matrix_c_contiguous.hpp"
#include "common/input/label_matrix_csr.hpp"
#include "common/indices/index_vector_complete.hpp"
#include "common/indices/index_vector_partial.hpp"


/**
 * Implements column-wise read-only access to the labels of individual training examples that are stored in a matrix in
 * the compressed sparse column (CSC) format.
 *
 * This class provides copy constructors for copying an existing `CContiguousLabelMatrix`, which provides random access,
 * or a `CsrLabelMatrix`, which provides row-wise access to the labels of the training examples. These constructors
 * expect the indices of the examples to be considered when copying the existing label matrix to be provided.
 */
class CscLabelMatrix final : public BinaryCscConstView {

    public:

        /**
         * @param labelMatrix   A reference to an object of type `CContiguousLabelMatrix` to be copied
         * @param indicesBegin  A `CompleteIndexVector::const_iterator` to the beginning of the indices of the examples
         *                      to be considered
         * @param indicesEnd    A `CompleteIndexVector::const_iterator` to the end of the indices of the examples to be
         *                      considered
         */
        CscLabelMatrix(const CContiguousLabelMatrix& labelMatrix, CompleteIndexVector::const_iterator indicesBegin,
                       CompleteIndexVector::const_iterator indicesEnd);

        /**
         * @param labelMatrix   A reference to an object of type `CContiguousLabelMatrix` to be copied
         * @param indicesBegin  A `PartialIndexVector::const_iterator` to the beginning of the indices of the examples
         *                      to be considered
         * @param indicesEnd    A `PartialIndexVector::const_iterator` to the end of the indices of the examples to be
         *                      considered
         */
        CscLabelMatrix(const CContiguousLabelMatrix& labelMatrix, PartialIndexVector::const_iterator indicesBegin,
                       PartialIndexVector::const_iterator indicesEnd);

        /**
         * @param labelMatrix   A reference to an object of type `CsrLabelMatrix` to be copied
         * @param indicesBegin  A `CompleteIndexVector::const_iterator` to the beginning of the indices of the examples
         *                      to be considered
         * @param indicesEnd    A `CompleteIndexVector::const_iterator` to the end of the indices of the examples to be
         *                      considered
         */
        CscLabelMatrix(const CsrLabelMatrix& labelMatrix, CompleteIndexVector::const_iterator indicesBegin,
                       CompleteIndexVector::const_iterator indicesEnd);

        /**
         * @param labelMatrix   A reference to an object of type `CsrLabelMatrix` to be copied
         * @param indicesBegin  A `PartialIndexVector::const_iterator` to the beginning of the indices of the examples
         *                      to be considered
         * @param indicesEnd    A `PartialIndexVector::const_iterator` to the end of the indices of the examples to be
         *                      considered
         */
        CscLabelMatrix(const CsrLabelMatrix& labelMatrix, PartialIndexVector::const_iterator indicesBegin,
                       PartialIndexVector::const_iterator indicesEnd);

        ~CscLabelMatrix();

};
