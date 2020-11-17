/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "label_matrix.h"


/**
 * Implements random access to the labels of the training examples based on a C-contiguous array.
 */
class DenseLabelMatrix : public IRandomAccessLabelMatrix {

    private:

        uint32 numExamples_;

        uint32 numLabels_;

        const uint8* y_;

    public:

        /**
         * @param numExamples   The number of examples
         * @param numLabels     The number of labels
         * @param y             A pointer to a C-contiguous array of type `uint8`, shape `(numExamples, numLabels)`,
         *                      representing the labels of the training examples
         */
        DenseLabelMatrix(uint32 numExamples, uint32 numLabels, const uint8* y);

        uint32 getNumExamples() const override;

        uint32 getNumLabels() const override;

        uint8 getValue(uint32 exampleIndex, uint32 labelIndex) const override;

};
