/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_matrix_c_contiguous.hpp"
#include "mlrl/common/data/view_matrix_csr.hpp"
#include "mlrl/common/data/view_matrix_csr_binary.hpp"
#include "mlrl/common/sampling/partition.hpp"
#include "mlrl/common/sampling/random.hpp"

#include <memory>

/**
 * Defines an interface for all classes that implement a method for partitioning the available training examples into a
 * training set and a holdout set.
 */
class IPartitionSampling {
    public:

        virtual ~IPartitionSampling() {}

        /**
         * Creates and returns a partition of the available training examples.
         *
         * @param rng   A reference to an object of type `RNG`, implementing the random number generator to be used
         * @return      A reference to an object of type `IPartition` that provides access to the indices of the
         *              training examples that belong to the training set and holdout set, respectively
         */
        virtual IPartition& partition(RNG& rng) = 0;
};

/**
 * Defines an interface for all factories that allow to create objects of type `IPartitionSampling` that can be used in
 * classification problems.
 */
class IClassificationPartitionSamplingFactory {
    public:

        virtual ~IClassificationPartitionSamplingFactory() {}

        /**
         * Creates and returns a new object of type `IPartitionSampling`.
         *
         * @param labelMatrix   A reference to an object of type `CContiguousView` that provides random access to the
         *                      labels of the training examples
         * @return              An unique pointer to an object of type `IPartitionSampling` that has been created
         */
        virtual std::unique_ptr<IPartitionSampling> create(const CContiguousView<const uint8>& labelMatrix) const = 0;

        /**
         * Creates and returns a new object of type `IPartitionSampling`.
         *
         * @param labelMatrix   A reference to an object of type `BinaryCsrView` that provides row-wise access to the
         *                      labels of the training examples
         * @return              An unique pointer to an object of type `IPartitionSampling` that has been created
         */
        virtual std::unique_ptr<IPartitionSampling> create(const BinaryCsrView& labelMatrix) const = 0;
};

/**
 * Defines an interface for all factories that allow to create objects of type `IPartitionSampling` that can be used in
 * regression problems.
 */
class IRegressionPartitionSamplingFactory {
    public:

        virtual ~IRegressionPartitionSamplingFactory() {}

        /**
         * Creates and returns a new object of type `IPartitionSampling`.
         *
         * @param regressionMatrix  A reference to an object of type `CContiguousView` that provides random access to
         *                          the regression scores of the training examples
         * @return                  An unique pointer to an object of type `IPartitionSampling` that has been created
         */
        virtual std::unique_ptr<IPartitionSampling> create(
          const CContiguousView<const float32>& regressionMatrix) const = 0;

        /**
         * Creates and returns a new object of type `IPartitionSampling`.
         *
         * @param regressionMatrix  A reference to an object of type `CsrView` that provides row-wise access to the
         *                          regression scores of the training examples
         * @return                  An unique pointer to an object of type `IPartitionSampling` that has been created
         */
        virtual std::unique_ptr<IPartitionSampling> create(const CsrView<const float32>& regressionMatrix) const = 0;
};

/**
 * Defines an interface for all classes that allow to configure a method for partitioning the available training
 * examples into a training set and a holdout set that can be used in classification problems.
 */
class IClassificationPartitionSamplingConfig {
    public:

        virtual ~IClassificationPartitionSamplingConfig() {}

        /**
         * Creates and returns a new object of type `IClassificationPartitionSamplingFactory` according to the specified
         * configuration.
         *
         * @return An unique pointer to an object of type `IClassificationPartitionSamplingFactory` that has been
         * created
         */
        virtual std::unique_ptr<IClassificationPartitionSamplingFactory> createClassificationPartitionSamplingFactory()
          const = 0;
};

/**
 * Defines an interface for all classes that allow to configure a method for partitioning the available training
 * examples into a training set and a holdout set that can be used in regression problems.
 */
class IRegressionPartitionSamplingConfig {
    public:

        virtual ~IRegressionPartitionSamplingConfig() {}

        /**
         * Creates and returns a new object of type `IRegressionPartitionSamplingFactory` according to the specified
         * configuration.
         *
         * @return An unique pointer to an object of type `IRegressionPartitionSamplingFactory` that has been
         * created
         */
        virtual std::unique_ptr<IRegressionPartitionSamplingFactory> createRegressionPartitionSamplingFactory()
          const = 0;
};
