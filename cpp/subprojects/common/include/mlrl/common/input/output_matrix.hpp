/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"
#include "mlrl/common/util/dll_exports.hpp"

/**
 * Defines an interface for all output matrices that store the ground truth to be used for training a model.
 */
class MLRLCOMMON_API IOutputMatrix {
    public:

        virtual ~IOutputMatrix() {}

        /**
         * Returns whether the output matrix is sparse or not.
         *
         * @return True, if the output matrix is sparse, false otherwise
         */
        virtual bool isSparse() const = 0;

        /**
         * Returns the number of examples in the output matrix.
         *
         * @return The number of examples
         */
        virtual uint32 getNumExamples() const = 0;

        /**
         * Returns the number of outputs in the output matrix.
         *
         * @return The number of outputs
         */
        virtual uint32 getNumOutputs() const = 0;
};
