/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/sampling/partition_sampling.hpp"
#include "mlrl/common/util/properties.hpp"

#include <memory>

/**
 * Defines an interface for all classes that allow to configure a method for partitioning the available training
 * examples into a training set and a holdout set using stratification, such that for each label the proportion of
 * relevant and irrelevant examples is maintained.
 */
class MLRLCOMMON_API IOutputWiseStratifiedBiPartitionSamplingConfig {
    public:

        virtual ~IOutputWiseStratifiedBiPartitionSamplingConfig() {}

        /**
         * Returns the fraction of examples that are included in the holdout set.
         *
         * @return The fraction of examples that are included in the holdout set
         */
        virtual float32 getHoldoutSetSize() const = 0;

        /**
         * Sets the fraction of examples that should be included in the holdout set.
         *
         * @param holdoutSetSize    The fraction of examples that should be included in the holdout set, e.g. a value of
         *                          0.6 corresponds to 60 % of the available examples. Must be in (0, 1)
         * @return                  A reference to an object of type `IOutputWiseStratifiedBiPartitionSamplingConfig`
         *                          that allows further configuration of the method for partitioning the available
         *                          training examples into a training set and a holdout set
         */
        virtual IOutputWiseStratifiedBiPartitionSamplingConfig& setHoldoutSetSize(float32 holdoutSetSize) = 0;
};

/**
 * Allows to configure a method for partitioning the available training examples into a training set and a holdout set
 * using stratification, such that for each label the proportion of relevant and irrelevant examples is maintained.
 */
class OutputWiseStratifiedBiPartitionSamplingConfig final : public IClassificationPartitionSamplingConfig,
                                                            public IOutputWiseStratifiedBiPartitionSamplingConfig {
    private:

        const ReadableProperty<RNGConfig> rngConfig_;

        float32 holdoutSetSize_;

    public:

        /**
         * @param rngConfig A `ReadableProperty` that provides access to the `RNGConfig` that stores the configuration
         *                  of random number generators
         */
        OutputWiseStratifiedBiPartitionSamplingConfig(ReadableProperty<RNGConfig> rngConfig);

        float32 getHoldoutSetSize() const override;

        IOutputWiseStratifiedBiPartitionSamplingConfig& setHoldoutSetSize(float32 holdoutSetSize) override;

        std::unique_ptr<IClassificationPartitionSamplingFactory> createClassificationPartitionSamplingFactory()
          const override;
};
