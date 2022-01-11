/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/sampling/partition_sampling.hpp"


/**
 * Allows to configure a method for partitioning the available training examples into a training set and a holdout set
 * that does not split the training examples, but includes all of them in the training set.
 */
class NoPartitionSamplingConfig final : public IPartitionSamplingConfig {

};

/**
 * Allows to create objects of the type `IPartitionSampling` that do not split the training examples, but include all of
 * them in the training set.
 */
class NoPartitionSamplingFactory final : public IPartitionSamplingFactory {

    public:

        std::unique_ptr<IPartitionSampling> create(const CContiguousLabelMatrix& labelMatrix) const override;

        std::unique_ptr<IPartitionSampling> create(const CsrLabelMatrix& labelMatrix) const override;

};
