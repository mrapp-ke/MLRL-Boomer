/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/sampling/partition_sampling.hpp"

#include <memory>

/**
 * Allows to configure a method for partitioning the available training examples into a training set and a holdout set
 * that does not split the training examples, but includes all of them in the training set.
 */
class NoPartitionSamplingConfig final : public IPartitionSamplingConfig {
    public:

        std::unique_ptr<IPartitionSamplingFactory> createPartitionSamplingFactory() const override;
};
