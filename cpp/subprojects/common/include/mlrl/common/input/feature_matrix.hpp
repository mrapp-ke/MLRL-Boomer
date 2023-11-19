/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"
#include "mlrl/common/util/dll_exports.hpp"

/**
 * Defines an interface for all feature matrices.
 */
class MLRLCOMMON_API IFeatureMatrix {
    public:

        virtual ~IFeatureMatrix() {}

        /**
         * Returns whether the feature matrix is sparse or not.
         *
         * @return True, if the feature matrix is sparse, false otherwise
         */
        virtual bool isSparse() const = 0;

        /**
         * Returns the number of examples in the feature matrix.
         *
         * @return The number of examples
         */
        virtual uint32 getNumExamples() const = 0;

        /**
         * Returns the number of features in the feature matrix.
         *
         * @return The number of features
         */
        virtual uint32 getNumFeatures() const = 0;
};
