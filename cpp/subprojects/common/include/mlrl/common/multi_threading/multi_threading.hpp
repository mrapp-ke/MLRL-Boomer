/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/feature_matrix.hpp"
#include "mlrl/common/input/output_matrix.hpp"

/**
 * Stores settings to be used by parallelizable algorithms.
 */
struct MultiThreadingSettings final {
    public:

        /**
         * @param numThreads The number of threads that can be used by parallelizable algorithms
         */
        MultiThreadingSettings(uint32 numThreads) : numThreads(numThreads) {}

        /**
         * The number of threats that can be used by parallelizable algorithms.
         */
        const uint32 numThreads;
};

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

        /**
         * Determines and returns the settings to be used by parallelizable algorithms, depending on the available
         * hardware and whether multi-threading support was enabled at compile-time.
         *
         * @param featureMatrix A reference to an object of type `IFeatureMatrix` that provides access to the feature
         *                      values of the training examples
         * @param numOutputs    The total number of available outputs
         * @return              The `MultiThreadingSettings` to be used
         */
        virtual MultiThreadingSettings getSettings(const IFeatureMatrix& featureMatrix, uint32 numOutputs) const = 0;
};
