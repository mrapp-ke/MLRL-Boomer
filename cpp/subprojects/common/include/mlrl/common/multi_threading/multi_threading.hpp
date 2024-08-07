/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/feature_matrix.hpp"

/**
 * Defines an interface for all classes that allow to configure the multi-threading behavior of a parallelizable
 * algorithm.
 */
class IMultiThreadingConfig {
    public:

        virtual ~IMultiThreadingConfig() {}

        /**
         * Determines and returns the number of threads that can actually be used by a parallelizable algorithm,
         * depending on the number of available CPU cores and whether multi-threading support was enabled at
         * compile-time.
         *
         * @param featureMatrix A reference to an object of type `IFeatureMatrix` that provides access to the feature
         *                      values of the training examples
         * @param numOutputs    The total number of available outputs
         * @return              The number of threads that can actually be used
         */
        virtual uint32 getNumThreads(const IFeatureMatrix& featureMatrix, uint32 numOutputs) const = 0;
};
