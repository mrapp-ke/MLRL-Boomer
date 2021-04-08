/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/input/label_vector.hpp"
#include <memory>


/**
 * Defines an interface for all label matrices that provide access to the labels of the training examples.
 */
class ILabelMatrix {

    public:

        virtual ~ILabelMatrix() { };

        /**
         * Returns the number of available examples.
         *
         * @return The number of examples
         */
        virtual uint32 getNumRows() const = 0;

        /**
         * Returns the number of available labels.
         *
         * @return The number of labels
         */
        virtual uint32 getNumCols() const = 0;

        /**
         * Creates and returns a label vector that corresponds to a specific row in the label matrix.
         *
         * @param row   The row
         * @return      An unique pointer to an object of type `LabelVector` that has been created
         */
        virtual std::unique_ptr<LabelVector> getLabelVector(uint32 row) const = 0;

};

/**
 * Defines an interface for all label matrices that provide random access to the labels of the training examples.
 */
class IRandomAccessLabelMatrix : public ILabelMatrix {

    public:

        virtual ~IRandomAccessLabelMatrix() { };

        /**
         * Returns the value of a specific label.
         *
         * @param exampleIndex  The index of the example
         * @param labelIndex    The index of the label
         * @return              The value of the label
         */
        virtual uint8 getValue(uint32 exampleIndex, uint32 labelIndex) const = 0;

};
