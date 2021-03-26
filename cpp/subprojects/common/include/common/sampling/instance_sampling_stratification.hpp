#pragma once
#include "common/data/types.hpp"
#include "common/sampling/instance_sampling.hpp"


class Stratification final : public IInstanceSubSampling {
    private:
        float32 sampleSize_;

    public:
        Stratification(float32 sampleSize);

        std::unique_ptr<IWeightVector> subSample(const SinglePartition& partition, RNG& rng, const IRandomAccessLabelMatrix& labelMatrix) const override;

        std::unique_ptr<IWeightVector> subSample(const BiPartition& partition, RNG& rng, const IRandomAccessLabelMatrix& labelMatrix) const override;

};