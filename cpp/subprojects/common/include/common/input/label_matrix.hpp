/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"


/**
 * Defines an interface for all label matrices.
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

};
