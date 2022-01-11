/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/sampling/label_sampling.hpp"


/**
 * Allows to configure a method for sampling labels without replacement.
 */
class LabelSamplingWithoutReplacementConfig : public ILabelSamplingConfig {

    private:

        uint32 numSamples_;

    public:

        LabelSamplingWithoutReplacementConfig();

        /**
         * Returns the number of labels that are included in a sample.
         *
         * @return The number of labels that are included in a sample
         */
        uint32 getNumSamples() const;

        /**
         * Sets the number of labels that should be included in a sample.
         *
         * @param numSamples    The number of labels that should be included in a sample. Must be at least 1
         * @return              A reference to an object of type `LabelSamplingWithoutReplacementConfig` that allows
         *                      further configuration of the method for sampling labels
         */
        LabelSamplingWithoutReplacementConfig& setNumSamples(uint32 numSamples);

};

/**
 * Allows to create objects of type `ILabelSampling` that select a random subset of the available features without
 * replacement.
 */
class LabelSamplingWithoutReplacementFactory final : public ILabelSamplingFactory {

    private:

        uint32 numSamples_;

    public:

        /**
         * @param numSamples The number of labels to be included in the sample. Must be at least 1
         */
        LabelSamplingWithoutReplacementFactory(uint32 numSamples);

        std::unique_ptr<ILabelSampling> create(uint32 numLabels) const override;

};
