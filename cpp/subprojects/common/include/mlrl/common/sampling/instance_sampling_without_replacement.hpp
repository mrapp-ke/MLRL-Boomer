/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/sampling/instance_sampling.hpp"
#include "mlrl/common/util/properties.hpp"

#include <memory>

/**
 * Defines an interface for all classes that allow to configure a method for selecting a subset of the available
 * training examples without replacement.
 */
class MLRLCOMMON_API IInstanceSamplingWithoutReplacementConfig {
    public:

        virtual ~IInstanceSamplingWithoutReplacementConfig() {}

        /**
         * Returns the fraction of examples that are included in a sample.
         *
         * @return The fraction of examples that are included in a sample
         */
        virtual float32 getSampleSize() const = 0;

        /**
         * Sets the fraction of examples that should be included in a sample.
         *
         * @param sampleSize    The fraction of examples that should be included in a sample, e.g., a value of 0.6
         *                      corresponds to 60 % of the available training examples. Must be in (0, 1)
         * @return              A reference to an object of type `IInstanceSamplingWithoutReplacementConfig` that allows
         *                      further configuration of the method for sampling instances
         */
        virtual IInstanceSamplingWithoutReplacementConfig& setSampleSize(float32 sampleSize) = 0;

        /**
         * Returns the minimum number of examples that are included in a sample.
         *
         * @return The minimum number of examples that are included in a sample
         */
        virtual uint32 getMinSamples() const = 0;

        /**
         * Sets the minimum number of examples that should be included in a sample.
         *
         * @param minSamples    The minimum number of examples that should be included in a sample. Must be at least 1
         * @return              A reference to an object of type `IInstanceSamplingWithoutReplacementConfig` that allows
         *                      further configuration of the method for sampling instances
         */
        virtual IInstanceSamplingWithoutReplacementConfig& setMinSamples(uint32 minSamples) = 0;

        /**
         * Returns the maximum number of examples that are included in a sample.
         *
         * @return The maximum number of examples that are included in a sample
         */
        virtual uint32 getMaxSamples() const = 0;

        /**
         * Sets the maximum number of examples that should be included in a sample.
         *
         * @param maxSamples    The maximum number of examples that should be included in a sample. Must be at the value
         *                      returned by `getMaxSamples` or 0, if the number of examples should not be restricted
         * @return              A reference to an object of type `IInstanceSamplingWithoutReplacementConfig` that allows
         *                      further configuration of the method for sampling instances
         */
        virtual IInstanceSamplingWithoutReplacementConfig& setMaxSamples(uint32 maxSamples) = 0;
};

/**
 * Allows to configure a method for selecting a subset of the available training examples without replacement.
 */
class InstanceSamplingWithoutReplacementConfig final : public IClassificationInstanceSamplingConfig,
                                                       public IRegressionInstanceSamplingConfig,
                                                       public IInstanceSamplingWithoutReplacementConfig {
    private:

        const ReadableProperty<RNGConfig> rngConfig_;

        float32 sampleSize_;

        uint32 minSamples_;

        uint32 maxSamples_;

    public:

        /**
         * @param rngConfig A `ReadableProperty` that provides access to the `RNGConfig` that stores the configuration
         *                  of random number generators
         */
        InstanceSamplingWithoutReplacementConfig(ReadableProperty<RNGConfig> rngConfig);

        float32 getSampleSize() const override;

        IInstanceSamplingWithoutReplacementConfig& setSampleSize(float32 sampleSize) override;

        uint32 getMinSamples() const override;

        IInstanceSamplingWithoutReplacementConfig& setMinSamples(uint32 minSamples) override;

        uint32 getMaxSamples() const override;

        IInstanceSamplingWithoutReplacementConfig& setMaxSamples(uint32 maxSamples) override;

        std::unique_ptr<IClassificationInstanceSamplingFactory> createClassificationInstanceSamplingFactory()
          const override;

        std::unique_ptr<IRegressionInstanceSamplingFactory> createRegressionInstanceSamplingFactory() const override;
};
