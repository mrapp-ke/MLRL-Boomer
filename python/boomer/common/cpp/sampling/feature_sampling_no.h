/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "feature_sampling.h"


/**
 * An implementation of the class `IFeatureSubSampling` that does not perform any sampling, but includes all features.
 */
class NoFeatureSubSampling : public IFeatureSubSampling {

    public:

        std::unique_ptr<IIndexVector> subSample(uint32 numFeatures, RNG& rng) const override;

};
