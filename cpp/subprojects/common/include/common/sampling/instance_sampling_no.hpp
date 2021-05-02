/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/sampling/instance_sampling.hpp"


/**
 * An implementation of the class `IInstanceSubSampling` that does not perform any sampling, but assigns equal weights
 * to all examples.
 */
class NoInstanceSubSampling final : public IInstanceSubSampling {

    public:

        std::unique_ptr<IWeightVector> subSample(const SinglePartition& partition, RNG& rng, const IRandomAccessLabelMatrix& labelMatrix, const IStatistics& statistics) const override;

        std::unique_ptr<IWeightVector> subSample(const BiPartition& partition, RNG& rng, const IRandomAccessLabelMatrix& labelMatrix, const IStatistics& statistics) const override;

};
