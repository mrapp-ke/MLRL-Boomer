/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../data/matrix_dok_binary.h"
#include "label_matrix.h"


/**
 * Implements random access to the labels of individual training examples that are stored in a pre-allocated sparse
 * matrix in the dictionary of keys (DOK) format.
 */
class DokLabelMatrix : public IRandomAccessLabelMatrix {

    private:

        uint32 numExamples_;

        uint32 numLabels_;

        BinaryDokMatrix matrix_;

    public:

        /**
         * @param numExamples   The number of examples
         * @param numLabels     The number of labels
         */
        DokLabelMatrix(uint32 numExamples, uint32 numLabels);

        /**
         * Marks a label of an example as relevant.
         *
         * @param exampleIndex  The index of the example
         * @param labelIndex    The index of the label
         */
        void setValue(uint32 exampleIndex, uint32 labelIndex);

        uint32 getNumRows() const override;

        uint32 getNumCols() const override;

        uint8 getValue(uint32 exampleIndex, uint32 labelIndex) const override;

};
