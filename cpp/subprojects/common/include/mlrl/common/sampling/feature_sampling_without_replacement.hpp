/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/sampling/feature_sampling.hpp"
#include "mlrl/common/util/properties.hpp"

#include <memory>

/**
 * Defines an interface for all classes that allow to configure a method for sampling features without replacement.
 */
class MLRLCOMMON_API IFeatureSamplingWithoutReplacementConfig {
    public:

        virtual ~IFeatureSamplingWithoutReplacementConfig() {}

        /**
         * Returns the fraction of features that are included in a sample.
         *
         * @return The fraction of features that are included in a sample
         */
        virtual float32 getSampleSize() const = 0;

        /**
         * Sets the fraction of features that should be included in a sample.
         *
         * @param sampleSize    The fraction of features that should be included in a sample, e.g., a value of 0.6
         *                      corresponds to 60 % of the available features. Must be in (0, 1) or 0, if the default
         *                      sample size `floor(log2(numFeatures - 1) + 1)` should be used
         * @return              A reference to an object of type `IFeatureSamplingWithoutReplacementConfig` that allows
         *                      further configuration of the method for sampling features
         */
        virtual IFeatureSamplingWithoutReplacementConfig& setSampleSize(float32 sampleSize) = 0;

        /**
         * Returns the minimum number of features that are included in a sample.
         *
         * @return The minimum number of features that are included in a sample
         */
        virtual uint32 getMinSamples() const = 0;

        /**
         * Sets the minimum number of features that should be included in a sample.
         *
         * @param minSamples    The minimum number of features that should be included in a sample. Must be at least 1
         * @return              A reference to an object of type `IFeatureSamplingWithoutReplacementConfig` that allows
         *                      further configuration of the method for sampling features
         */
        virtual IFeatureSamplingWithoutReplacementConfig& setMinSamples(uint32 minSamples) = 0;

        /**
         * Returns the maximum number of features that are included in a sample.
         *
         * @return The maximum number of features that are included in a sample
         */
        virtual uint32 getMaxSamples() const = 0;

        /**
         * Sets the maximum number of features that should be included in a sample.
         *
         * @param maxSamples    The maximum number of features that should be included in a sample. Must be at the value
         *                      returned by `getMaxSamples` or 0, if the number of features should not be restricted
         * @return              A reference to an object of type `IFeatureSamplingWithoutReplacementConfig` that allows
         *                      further configuration of the method for sampling features
         */
        virtual IFeatureSamplingWithoutReplacementConfig& setMaxSamples(uint32 maxSamples) = 0;

        /**
         * Returns the number of trailing features that are always included in a sample.
         *
         * @return The number of trailing features that are always included in a sample
         */
        virtual uint32 getNumRetained() const = 0;

        /**
         * Sets the number fo trailing features that should always be included in a sample.
         *
         * @param numRetained   The number of trailing features that should always be included in a sample. Must be at
         *                      least 0
         * @return              A reference to an object of type `IFeatureSamplingWithoutReplacementConfig` that allows
         *                      further configuration of the method for sampling features
         */
        virtual IFeatureSamplingWithoutReplacementConfig& setNumRetained(uint32 numRetained) = 0;
};

/**
 * Allows to configure a method for sampling features without replacement.
 */
class FeatureSamplingWithoutReplacementConfig final : public IFeatureSamplingConfig,
                                                      public IFeatureSamplingWithoutReplacementConfig {
    private:

        const ReadableProperty<RNGConfig> rngConfig_;

        float32 sampleSize_;

        uint32 minSamples_;

        uint32 maxSamples_;

        uint32 numRetained_;

    public:

        /**
         * @param rngConfig A `ReadableProperty` that allows to access the `RNGConfig` that stores the configuration of
         *                  random number generators
         */
        FeatureSamplingWithoutReplacementConfig(ReadableProperty<RNGConfig> rngConfig);

        float32 getSampleSize() const override;

        IFeatureSamplingWithoutReplacementConfig& setSampleSize(float32 sampleSize) override;

        uint32 getMinSamples() const override;

        IFeatureSamplingWithoutReplacementConfig& setMinSamples(uint32 minSamples) override;

        uint32 getMaxSamples() const override;

        IFeatureSamplingWithoutReplacementConfig& setMaxSamples(uint32 maxSamples) override;

        uint32 getNumRetained() const override;

        IFeatureSamplingWithoutReplacementConfig& setNumRetained(uint32 numRetained) override;

        std::unique_ptr<IFeatureSamplingFactory> createFeatureSamplingFactory(
          const IFeatureMatrix& featureMatrix) const override;

        bool isSamplingUsed() const override;
};
