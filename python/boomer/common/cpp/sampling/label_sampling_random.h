/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "label_sampling.h"


/**
 * Implements random label subset selection for selecting a random subset of the available features without replacement.
 */
class RandomLabelSubsetSelection : public ILabelSubSampling {

    private:

        uint32 numSamples_;

    public:

        /**
         * @param The number of labels to be included in the sample
         */
        RandomLabelSubsetSelection(uint32 numSamples);

        std::unique_ptr<IIndexVector> subSample(uint32 numLabels, RNG& rng) const override;

};
