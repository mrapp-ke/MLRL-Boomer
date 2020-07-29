/**
 * Implements classes that provide access to the labels of training examples.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "arrays.h"


namespace statistics {

    /**
     * An abstract base class for all label matrices that provide random access to the labels of the training examples.
     */
    class AbstractLabelMatrix {

        public:

            /**
             * Creates a new label matrix.
             *
             * @param numExamples   The number of examples
             * @param numLabels     The number of labels
             */
            AbstractLabelMatrix(intp numExamples, intp numLabels);

            /**
             * Frees the memory occupied by the label matrix.
             */
            virtual ~AbstractLabelMatrix();

            /**
             * The number of examples.
             */
            intp numExamples_;

            /**
             * The number of labels.
             */
            intp numLabels_;

            /**
             * Returns whether a specific label of the example at a given index is relevant or irrelevant.
             *
             * @param exampleIndex  The index of the example
             * @param labelIndex    The index of the label
             * @return              1, if the label is relevant, 0 otherwise
             */
            virtual uint8 getLabel(intp exampleIndex, intp labelIndex);

    };

    /**
     * Implements random access to the labels of the training examples based on a C-contiguous array.
     */
    class DenseLabelMatrixImpl : public AbstractLabelMatrix {

        public:

            /**
             * Creates a new label matrix that provides random access to the labels of the training examples based on a
             * C-contiguous array.
             *
             * @param numExamples   The number of examples
             * @param numLabels     The number of labels
             */
            DenseLabelMatrixImpl(intp numExamples, intp numLabels);

            ~DenseLabelMatrixImpl();

            uint8 getLabel(intp exampleIndex, intp labelIndex) override;

    };

    /**
     * Implements random access to the labels of the training examples based on a sparse matrix in the dictionary of
     * keys (DOK) format.
     */
    class DokLabelMatrixImpl : public AbstractLabelMatrix {

        public:

            /**
             * Creates a new label matrix that provides random access to the labels of the training examples based on a
             * sparse matrix in the DOK format.
             *
             * @param numExamples   The number of examples
             * @param numLabels     The number of labels
             */
            DokLabelMatrixImpl(intp numExamples, intp numLabels);

            ~DokLabelMatrixImpl();

            uint8 getLabel(intp exampleIndex, intp labelIndex) override;

    };

}
