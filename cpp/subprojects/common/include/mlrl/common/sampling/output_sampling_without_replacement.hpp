/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/sampling/output_sampling.hpp"
#include "mlrl/common/util/properties.hpp"

#include <memory>

/**
 * Defines an interface for all classes that allow to configure a method for sampling outputs without replacement.
 */
class MLRLCOMMON_API IOutputSamplingWithoutReplacementConfig {
    public:

        virtual ~IOutputSamplingWithoutReplacementConfig() {}

        /**
         * Returns the fraction of outputs that are included in a sample.
         *
         * @return The fraction of outputs that are included in a sample
         */
        virtual float32 getSampleSize() const = 0;

        /**
         * Sets the fraction of outputs that should be included in a sample.
         *
         * @param sampleSize    The fraction of outputs that should be included in a sample, e.g., a value of 0.6
         *                      corresponds to 60 % of the available outputs. Must be in (0, 1)
         * @return              A reference to an object of type `IOutputSamplingWithoutReplacementConfig` that allows
         *                      further configuration of the method for sampling instances
         */
        virtual IOutputSamplingWithoutReplacementConfig& setSampleSize(float32 sampleSize) = 0;

        /**
         * Returns the minimum number of outputs that are included in a sample.
         *
         * @return The minimum number of outputs that are included in a sample
         */
        virtual uint32 getMinSamples() const = 0;

        /**
         * Sets the minimum number of outputs that should be included in a sample.
         *
         * @param minSamples    The minimum number of outputs that should be included in a sample. Must be at least 1
         * @return              A reference to an object of type `IOutputSamplingWithoutReplacementConfig` that allows
         *                      further configuration of the method for sampling instances
         */
        virtual IOutputSamplingWithoutReplacementConfig& setMinSamples(uint32 minSamples) = 0;

        /**
         * Returns the maximum number of outputs that are included in a sample.
         *
         * @return The maximum number of outputs that are included in a sample
         */
        virtual uint32 getMaxSamples() const = 0;

        /**
         * Sets the maximum number of outputs that should be included in a sample.
         *
         * @param maxSamples    The maximum number of outputs that should be included in a sample. Must be at the value
         *                      returned by `getMaxSamples` or 0, if the number of outputs should not be restricted
         * @return              A reference to an object of type `IOutputSamplingWithoutReplacementConfig` that allows
         *                      further configuration of the method for sampling instances
         */
        virtual IOutputSamplingWithoutReplacementConfig& setMaxSamples(uint32 maxSamples) = 0;
};

/**
 * Allows to configure a method for sampling outputs without replacement.
 */
class OutputSamplingWithoutReplacementConfig final : public IOutputSamplingConfig,
                                                     public IOutputSamplingWithoutReplacementConfig {
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
        OutputSamplingWithoutReplacementConfig(ReadableProperty<RNGConfig> rngConfig);

        float32 getSampleSize() const override;

        IOutputSamplingWithoutReplacementConfig& setSampleSize(float32 sampleSize) override;

        uint32 getMinSamples() const override;

        IOutputSamplingWithoutReplacementConfig& setMinSamples(uint32 minSamples) override;

        uint32 getMaxSamples() const override;

        IOutputSamplingWithoutReplacementConfig& setMaxSamples(uint32 maxSamples) override;

        std::unique_ptr<IOutputSamplingFactory> createOutputSamplingFactory(
          const IOutputMatrix& outputMatrix) const override;
};
