/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/sampling/partition_sampling.hpp"


/**
 * Allows to configure a method for partitioning the available training examples into a training set and a holdout set
 * that randomly split the training examples into two mutually exclusive sets.
 */
class RandomBiPartitionSamplingConfig : public IPartitionSamplingConfig {

    private:

        float32 holdoutSetSize_;

    public:

        RandomBiPartitionSamplingConfig();

        /**
         * Returns the fraction of examples that are included in the holdout set.
         *
         * @return The fraction of examples that are included in the holdout set
         */
        float32 getHoldoutSetSize() const;

        /**
         * Sets the fraction of examples that should be included in the holdout set.
         *
         * @param holdoutSetSize    The fraction of examples that should be included in the holdout set, e.g. a value of
         *                          0.6 corresponds to 60 % of the available examples. Must be in (0, 1)
         * @return                  A reference to an object of type `RandomBiPartitionSamplingConfig` that allows
         *                          further configuration of the method for partitioning the available training examples
         *                          into a training set and a holdout set
         */
        RandomBiPartitionSamplingConfig& setHoldoutSetSize(float32 holdoutSetSize);

};

/**
 * Allows to create objects of the type `IPartitionSampling` that randomly split the training examples into two mutually
 * exclusive sets that may be used as a training set and a holdout set.
 */
class RandomBiPartitionSamplingFactory final : public IPartitionSamplingFactory {

    private:

        float32 holdoutSetSize_;

    public:

        /**
         * @param holdoutSetSize The fraction of examples to be included in the holdout set (e.g. a value of 0.6
         *                       corresponds to 60 % of the available examples). Must be in (0, 1)
         */
        RandomBiPartitionSamplingFactory(float32 holdoutSetSize);

        std::unique_ptr<IPartitionSampling> create(const CContiguousLabelMatrix& labelMatrix) const override;

        std::unique_ptr<IPartitionSampling> create(const CsrLabelMatrix& labelMatrix) const override;

};
