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
             * Frees the memory occupied by the label matrix.
             */
            ~AbstractLabelMatrix();

            /**
             * Returns whether a specific label of the example at a given index is relevant or irrelevant.
             *
             * @param exampleIndex  The index of the example
             * @param labelIndex    The index of the label
             * @return              1, if the label is relevant, 0 otherwise
             */
            uint8 getLabel(intp exampleIndex, intp labelIndex);

    };

}
