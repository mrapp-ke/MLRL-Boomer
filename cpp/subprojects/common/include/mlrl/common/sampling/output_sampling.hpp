/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/indices/index_vector.hpp"
#include "mlrl/common/input/output_matrix.hpp"
#include "mlrl/common/random/rng.hpp"

#include <memory>

/**
 * Defines an interface for all classes that implement a method for sampling outputs.
 */
class IOutputSampling {
    public:

        virtual ~IOutputSampling() {}

        /**
         * Creates and returns a sample of the available outputs.
         *
         * @return A reference to an object of type `IIndexVector` that provides access to the indices of the outputs
         *         that are contained in the sample
         */
        virtual const IIndexVector& sample() = 0;
};

/**
 * Defines an interface for all factories that allow to create objects of type `IOutputSampling`.
 */
class IOutputSamplingFactory {
    public:

        virtual ~IOutputSamplingFactory() {}

        /**
         * Creates and returns a new object of type `IOutputSampling`.
         *
         * @return An unique pointer to an object of type `IOutputSampling` that has been created
         */
        virtual std::unique_ptr<IOutputSampling> create() const = 0;
};

/**
 * Defines an interface for all classes that allow to configure a method for sampling outputs.
 */
class IOutputSamplingConfig {
    public:

        virtual ~IOutputSamplingConfig() {}

        /**
         * Creates and returns a new object of type `IOutputSamplingFactory` according to the specified configuration.
         *
         * @param outputMatrix  A reference to an object of type `IOutputMatrix` that provides access to the ground
         *                      truth of the training examples
         * @return              An unique pointer to an object of type `IOutputSamplingFactory` that has been created
         */
        virtual std::unique_ptr<IOutputSamplingFactory> createOutputSamplingFactory(
          const IOutputMatrix& outputMatrix) const = 0;
};
