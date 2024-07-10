/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/output_matrix.hpp"

#include <memory>

// Forward declarations
class IStatisticsProvider;
class IStatisticsProviderFactory;
class IPartitionSampling;
class IPartitionSamplingFactory;
class IInstanceSampling;
class IInstanceSamplingFactory;
class IStatistics;
class SinglePartition;
class BiPartition;

/**
 * Defines an interface for all output matrices that provide access to the ground truth of training examples.
 */
class MLRLCOMMON_API IRowWiseOutputMatrix : public IOutputMatrix {
    public:

        virtual ~IRowWiseOutputMatrix() override {}

        /**
         * Creates and returns a new instance of the class `IStatisticsProvider`, based on the type of this output
         * matrix.
         *
         * @param factory   A reference to an object of type `IStatisticsProviderFactory` that should be used to create
         *                  the instance
         * @return          An unique pointer to an object of type `IStatisticsProvider` that has been created
         */
        virtual std::unique_ptr<IStatisticsProvider> createStatisticsProvider(
          const IStatisticsProviderFactory& factory) const = 0;

        /**
         * Creates and returns a new instance of the class `IPartitionSampling`, based on the type of this output
         * matrix.
         *
         * @param factory   A reference to an object of type `IPartitionSamplingFactory` that should be used to create
         *                  the instance
         * @return          An unique pointer to an object of type `IPartitionSampling` that has been created
         */
        virtual std::unique_ptr<IPartitionSampling> createPartitionSampling(
          const IPartitionSamplingFactory& factory) const = 0;

        /**
         * Creates and returns a new instance of the class `IInstanceSampling`, based on the type of this output matrix.
         *
         * @param factory       A reference to an object of type `IInstanceSamplingFactory` that should be used to
         *                      create the instance
         * @param partition     A reference to an object of type `SinglePartition` that provides access to the indices
         *                      of the training examples that are included in the training set
         * @param statistics    A reference to an object of type `IStatistics` that provides access to statistics about
         *                      the quality of predictions for training examples
         * @return              An unique pointer to an object of type `IInstanceSampling` that has been created
         */
        virtual std::unique_ptr<IInstanceSampling> createInstanceSampling(const IInstanceSamplingFactory& factory,
                                                                          const SinglePartition& partition,
                                                                          IStatistics& statistics) const = 0;

        /**
         * Creates and returns a new instance of the class `IInstanceSampling`, based on the type of this output matrix.
         *
         * @param factory       A reference to an object of type `IInstanceSamplingFactory` that should be used to
         *                      create the instance
         * @param partition     A reference to an object of type `BiPartition` that provides access to the indices of
         *                      the training examples that are included in the training set and the holdout set,
         *                      respectively
         * @param statistics    A reference to an object of type `IStatistics` that provides access to statistics about
         *                      the quality of predictions for training examples
         * @return              An unique pointer to an object of type `IInstanceSampling` that has been created
         */
        virtual std::unique_ptr<IInstanceSampling> createInstanceSampling(const IInstanceSamplingFactory& factory,
                                                                          BiPartition& partition,
                                                                          IStatistics& statistics) const = 0;
};
