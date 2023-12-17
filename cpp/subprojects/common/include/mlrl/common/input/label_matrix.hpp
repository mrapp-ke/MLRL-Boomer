/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"
#include "mlrl/common/util/dll_exports.hpp"

/**
 * Defines an interface for all label matrices.
 */
class MLRLCOMMON_API ILabelMatrix {
    public:

        virtual ~ILabelMatrix() {}

        /**
         * Returns whether the label matrix is sparse or not.
         *
         * @return True, if the label matrix is sparse, false otherwise
         */
        virtual bool isSparse() const = 0;

        /**
         * Returns the number of examples in the label matrix.
         *
         * @return The number of examples
         */
        virtual uint32 getNumExamples() const = 0;

        /**
         * Returns the number of labels in the label matrix.
         *
         * @return The number of labels
         */
        virtual uint32 getNumLabels() const = 0;
};
