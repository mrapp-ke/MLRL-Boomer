/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/sampling/feature_sampling.hpp"


/**
 * An implementation of the class `IFeatureSubSampling` that does not perform any sampling, but includes all features.
 */
class NoFeatureSubSampling final : public IFeatureSubSampling {

    private:

        uint32 numFeatures_;

    public:

        /**
         * @param numFeatures The total number of available features
         */
        NoFeatureSubSampling(uint32 numFeatures);

        std::unique_ptr<IIndexVector> subSample(RNG& rng) const override;

};

/**
 * Allows to create instances of the type `IFeatureSubSampling` that do not perform any sampling, but include all
 * features.
 */
class NoFeatureSubSamplingFactory final : public IFeatureSubSamplingFactory {

    public:

        std::unique_ptr<IFeatureSubSampling> create(uint32 numFeatures) const override;

};
