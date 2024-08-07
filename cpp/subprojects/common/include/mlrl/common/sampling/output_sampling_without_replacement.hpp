/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/sampling/output_sampling.hpp"

#include <memory>

/**
 * Defines an interface for all classes that allow to configure a method for sampling outputs without replacement.
 */
class MLRLCOMMON_API IOutputSamplingWithoutReplacementConfig {
    public:

        virtual ~IOutputSamplingWithoutReplacementConfig() {}

        /**
         * Returns the number of outputs that are included in a sample.
         *
         * @return The number of outputs that are included in a sample
         */
        virtual uint32 getNumSamples() const = 0;

        /**
         * Sets the number of outputs that should be included in a sample.
         *
         * @param numSamples    The number of outputs that should be included in a sample. Must be at least 1
         * @return              A reference to an object of type `IOutputSamplingWithoutReplacementConfig` that allows
         *                      further configuration of the sampling method
         */
        virtual IOutputSamplingWithoutReplacementConfig& setNumSamples(uint32 numSamples) = 0;
};

/**
 * Allows to configure a method for sampling outputs without replacement.
 */
class OutputSamplingWithoutReplacementConfig final : public IOutputSamplingConfig,
                                                     public IOutputSamplingWithoutReplacementConfig {
    private:

        uint32 numSamples_;

    public:

        OutputSamplingWithoutReplacementConfig();

        uint32 getNumSamples() const override;

        IOutputSamplingWithoutReplacementConfig& setNumSamples(uint32 numSamples) override;

        std::unique_ptr<IOutputSamplingFactory> createOutputSamplingFactory(
          const IOutputMatrix& outputMatrix) const override;
};
