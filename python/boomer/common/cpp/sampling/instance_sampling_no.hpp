/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "instance_sampling.hpp"


/**
 * An implementation of the class `IInstanceSubSampling` that does not perform any sampling, but assigns equal weights
 * to all examples.
 */
class NoInstanceSubSampling final : public IInstanceSubSampling {

    public:

        std::unique_ptr<IWeightVector> subSample(uint32 numExamples, RNG& rng) const override;

};
