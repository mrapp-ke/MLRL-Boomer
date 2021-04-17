/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/sampling/partition_sampling.hpp"


/**
 * An implementation of the class `IPartitionSampling` that does not split the training examples, but includes all of
 * them in the training set.
 */
class NoPartitionSampling final : public IPartitionSampling {

    private:

        uint32 numExamples_;

    public:

        NoPartitionSampling(uint32 numExamples);

        std::unique_ptr<IPartition> createPartition(RNG& rng) const override;

};

/**
 * Allows to create objects of the type `IPartitionSampling` that do not split the training examples, but include all of
 * them in the training set.
 */
class NoPartitionSamplingFactory final : public IPartitionSamplingFactory {

    public:

        std::unique_ptr<IPartitionSampling> create(uint32 numExamples) const override;

};
