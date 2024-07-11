/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/output_matrix.hpp"

#include <memory>

// Forward declarations
class IInstanceSampling;
class IRegressionInstanceSamplingFactory;
class IStatistics;
class SinglePartition;
class BiPartition;
class IPartitionSampling;
class IRegressionPartitionSamplingFactory;
class IStatisticsProvider;
class IRegressionStatisticsProviderFactory;

/**
 * Defines an interface for all regression matrices that provide access to the ground truth regression scores of
 * training examples.
 */
class MLRLCOMMON_API IRowWiseRegressionMatrix : public IOutputMatrix {
    public:

        virtual ~IRowWiseRegressionMatrix() override {}

        /**
         * Creates and returns a new instance of the class `IInstanceSampling`, based on the type of this regression
         * matrix.
         *
         * @param factory       A reference to an object of type `IClassificationInstanceSamplingFactory` that should be
         *                       used to create the instance
         * @param partition     A reference to an object of type `SinglePartition` that provides access to the indices
         *                      of the training examples that are included in the training set
         * @param statistics    A reference to an object of type `IStatistics` that provides access to statistics about
         *                      the quality of predictions for training examples
         * @return              An unique pointer to an object of type `IInstanceSampling` that has been created
         */
        virtual std::unique_ptr<IInstanceSampling> createInstanceSampling(
          const IRegressionInstanceSamplingFactory& factory, const SinglePartition& partition,
          IStatistics& statistics) const = 0;

        /**
         * Creates and returns a new instance of the class `IInstanceSampling`, based on the type of this regression
         * matrix.
         *
         * @param factory       A reference to an object of type `IRegressionInstanceSamplingFactory` that should be
         *                      used to create the instance
         * @param partition     A reference to an object of type `BiPartition` that provides access to the indices of
         *                      the training examples that are included in the training set and the holdout set,
         *                      respectively
         * @param statistics    A reference to an object of type `IStatistics` that provides access to statistics about
         *                      the quality of predictions for training examples
         * @return              An unique pointer to an object of type `IInstanceSampling` that has been created
         */
        virtual std::unique_ptr<IInstanceSampling> createInstanceSampling(
          const IRegressionInstanceSamplingFactory& factory, BiPartition& partition, IStatistics& statistics) const = 0;

        /**
         * Creates and returns a new instance of the class `IPartitionSampling`, based on the type of this regression
         * matrix.
         *
         * @param factory   A reference to an object of type `IRegressionPartitionSamplingFactory` that should be used
         *                  to create the instance
         * @return          An unique pointer to an object of type `IPartitionSampling` that has been created
         */
        virtual std::unique_ptr<IPartitionSampling> createPartitionSampling(
          const IRegressionPartitionSamplingFactory& factory) const = 0;

        /**
         * Creates and returns a new instance of the class `IStatisticsProvider`, based on the type of this output
         * matrix.
         *
         * @param factory   A reference to an object of type `IRegressionStatisticsProviderFactory` that should be used
         *                  to create the instance
         * @return          An unique pointer to an object of type `IStatisticsProvider` that has been created
         */
        virtual std::unique_ptr<IStatisticsProvider> createStatisticsProvider(
          const IRegressionStatisticsProviderFactory& factory) const = 0;
};
