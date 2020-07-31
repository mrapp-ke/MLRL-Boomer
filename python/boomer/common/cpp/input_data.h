/**
 * Implements classes that provide access to the data that is provided for training.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "arrays.h"
#include "sparse.h"


/**
 * An abstract base class for all label matrices that provide random access to the labels of the training examples.
 */
class AbstractLabelMatrix {

    public:

        /**
         * @param numExamples   The number of examples
         * @param numLabels     The number of labels
         */
        AbstractLabelMatrix(intp numExamples, intp numLabels);

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

    private:

        const uint8* y_;

    public:

        /**
         * @param numExamples   The number of examples
         * @param numLabels     The number of labels
         * @param y             A pointer to a C-contiguous array of type `uint8`, shape `(numExamples, numLabels)`,
         *                      representing the labels of the training examples
         */
        DenseLabelMatrixImpl(intp numExamples, intp numLabels, const uint8* y);

        ~DenseLabelMatrixImpl();

        uint8 getLabel(intp exampleIndex, intp labelIndex) override;

};

/**
 * Implements random access to the labels of the training examples based on a sparse matrix in the dictionary of
 * keys (DOK) format.
 */
class DokLabelMatrixImpl : public AbstractLabelMatrix {

    private:

        BinaryDokMatrix* dokMatrix_;

    public:

        /**
         * @param numExamples   The number of examples
         * @param numLabels     The number of labels
         * @param dokMatrix     A pointer to an object of type `BinaryDokMatrix`, storing the relevant labels of the
         *                      training examples
         */
        DokLabelMatrixImpl(intp numExamples, intp numLabels, BinaryDokMatrix* dokMatrix);

        ~DokLabelMatrixImpl();

        uint8 getLabel(intp exampleIndex, intp labelIndex) override;

};
