/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/sampling/feature_sampling.hpp"


/**
 * Allows to create instances of the type `IFeatureSubSampling` that select a random subset of the available features
 * without replacement.
 */
class RandomFeatureSubsetSelectionFactory final : public IFeatureSubSamplingFactory {

    private:

        float32 sampleSize_;

    public:

        /**
         * @param sampleSize The fraction of features to be included in the sample (e.g. a value of 0.6 corresponds to
         *                   60 % of the available features). Must be in (0, 1) or 0, if the default sample size
         *                   `floor(log2(num_features - 1) + 1)` should be used
         */
        RandomFeatureSubsetSelectionFactory(float32 sampleSize);

        std::unique_ptr<IFeatureSubSampling> create(uint32 numFeatures) const override;

};
