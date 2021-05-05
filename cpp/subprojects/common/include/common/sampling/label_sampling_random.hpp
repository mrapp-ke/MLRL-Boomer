/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/sampling/label_sampling.hpp"


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
