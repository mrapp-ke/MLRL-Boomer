/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/sampling/feature_sampling.hpp"


/**
 * Allows to create instances of the type `IFeatureSubSampling` that do not perform any sampling, but include all
 * features.
 */
class NoFeatureSubSamplingFactory final : public IFeatureSubSamplingFactory {

    public:

        std::unique_ptr<IFeatureSubSampling> create(uint32 numFeatures) const override;

};
