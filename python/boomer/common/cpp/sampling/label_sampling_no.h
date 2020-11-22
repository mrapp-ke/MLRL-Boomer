/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "label_sampling.h"


/**
 * An implementation of the class `ILabelSubSampling` that does not perform any sampling, but includes all labels.
 */
class NoLabelSubSampling : public ILabelSubSampling {

    public:

        std::unique_ptr<IIndexVector> subSample(uint32 numLabels, RNG& rng) const override;

};
