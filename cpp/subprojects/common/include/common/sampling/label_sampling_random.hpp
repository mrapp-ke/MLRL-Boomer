/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/sampling/label_sampling.hpp"


/**
 * Implements random label subset selection for selecting a random subset of the available features without replacement.
 */
class RandomLabelSubsetSelection final : public ILabelSubSampling {

    private:

        uint32 numLabels_;

        uint32 numSamples_;

    public:

        /**
         * @param numLabels     The total number of available labels
         * @param numSamples    The number of labels to be included in the sample
         */
        RandomLabelSubsetSelection(uint32 numLabels, uint32 numSamples);

        std::unique_ptr<IIndexVector> subSample(RNG& rng) const override;

};

/**
 * Allows to create objects of type `ILabelSubSampling` that select a random subset of the available features without
 * replacement.
 */
class RandomLabelSubsetSelectionFactory final : public ILabelSubSamplingFactory {

    private:

        uint32 numSamples_;

    public:

        /**
         * @param numSamples The number of labels to be included in the sample
         */
        RandomLabelSubsetSelectionFactory(uint32 numSamples);

        std::unique_ptr<ILabelSubSampling> create(uint32 numLabels) const override;

};
