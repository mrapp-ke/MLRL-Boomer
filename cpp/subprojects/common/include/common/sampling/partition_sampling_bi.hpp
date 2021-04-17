/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/sampling/partition_sampling.hpp"


/**
 * An implementation of the class `IPartitionSampling` that splits the training examples into two mutually exclusive
 * sets that may be used as a training set and a holdout set.
 */
class BiPartitionSampling final : public IPartitionSampling {

    private:

        uint32 numHoldout_;

        uint32 numTraining_;

    public:

        /**
         * @param numExamples       The total number of available training examples
         * @param holdoutSetSize    The fraction of examples to be included in the holdout set (e.g. a value of 0.6
         *                          corresponds to 60 % of the available examples). Must be in (0, 1)
         */
        BiPartitionSampling(uint32 numExamples, float32 holdoutSetSize);

        std::unique_ptr<IPartition> partition(RNG& rng) const override;

};

/**
 * Allows to create objects of the type `IPartitionSampling` that split the training examples into two mutually
 * exclusive sets that may be used as a training set and a holdout set.
 */
class BiPartitionSamplingFactory final : public IPartitionSamplingFactory {

    private:

        float32 holdoutSetSize_;

    public:

        /**
         * @param holdoutSetSize The fraction of examples to be included in the holdout set (e.g. a value of 0.6
         *                       corresponds to 60 % of the available examples). Must be in (0, 1)
         */
        BiPartitionSamplingFactory(float32 holdoutSetSize);

        std::unique_ptr<IPartitionSampling> create(uint32 numExamples) const override;

};
