/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_matrix_c_contiguous.hpp"
#include "mlrl/common/data/view_matrix_csr.hpp"
#include "mlrl/common/data/view_matrix_csr_binary.hpp"
#include "mlrl/common/random/rng.hpp"
#include "mlrl/common/sampling/weight_vector.hpp"
#include "mlrl/common/statistics/statistics.hpp"

#include <memory>

// Forward declarations
class BiPartition;
class SinglePartition;

/**
 * Defines an interface for all classes that implement a method for sampling training examples.
 */
class IInstanceSampling {
    public:

        virtual ~IInstanceSampling() {}

        /**
         * Creates and returns a sample of the available training examples.
         *
         * @return A reference to an object type `WeightVector` that provides access to the weights of the individual
         *         training examples
         */
        virtual const IWeightVector& sample() = 0;
};

/**
 * Defines an interface for all factories that allow to create instances of the type `IInstanceSampling` that can be
 * used in classification problems.
 */
class IClassificationInstanceSamplingFactory {
    public:

        virtual ~IClassificationInstanceSamplingFactory() {}

        /**
         * Creates and returns a new object of type `IInstanceSampling`.
         *
         * @param labelMatrix   A reference to an object of type `CContiguousView` that provides access to the labels of
         *                      the training examples
         * @param partition     A reference to an object of type `SinglePartition` that provides access to the indices
         *                      of the training examples that are included in the training set
         * @param statistics    A reference to an object of type `IStatistics` that provides access to the statistics
         *                      which serve as a basis for learning rules
         * @return              An unique pointer to an object of type `IInstanceSampling` that has been created
         */
        virtual std::unique_ptr<IInstanceSampling> create(const CContiguousView<const uint8>& labelMatrix,
                                                          const SinglePartition& partition,
                                                          IStatistics& statistics) const = 0;

        /**
         * Creates and returns a new object of type `IInstanceSampling`.
         *
         * @param labelMatrix   A reference to an object of type `CContiguousView` that provides access to the labels of
         *                      the training examples
         * @param partition     A reference to an object of type `BiPartition` that provides access to the indices of
         *                      the training examples that are included in the training set and the holdout set,
         *                      respectively
         * @param statistics    A reference to an object of type `IStatistics` that provides access to the statistics
         *                      which serve as a basis for learning rules
         * @return              An unique pointer to an object of type `IInstanceSampling` that has been created
         */
        virtual std::unique_ptr<IInstanceSampling> create(const CContiguousView<const uint8>& labelMatrix,
                                                          BiPartition& partition, IStatistics& statistics) const = 0;

        /**
         * Creates and returns a new object of type `IInstanceSampling`.
         *
         * @param labelMatrix   A reference to an object of type `BinaryCsrView` that provides access to the labels of
         *                      the training examples
         * @param partition     A reference to an object of type `SinglePartition` that provides access to the indices
         *                      of the training examples that are included in the training set
         * @param statistics    A reference to an object of type `IStatistics` that provides access to the statistics
         *                      which serve as a basis for learning rules
         * @return              An unique pointer to an object of type `IInstanceSampling` that has been created
         */
        virtual std::unique_ptr<IInstanceSampling> create(const BinaryCsrView& labelMatrix,
                                                          const SinglePartition& partition,
                                                          IStatistics& statistics) const = 0;

        /**
         * Creates and returns a new object of type `IInstanceSampling`.
         *
         * @param labelMatrix   A reference to an object of type `BinaryCsrView` that provides access to the labels of
         *                      the training examples
         * @param partition     A reference to an object of type `BiPartition` that provides access to the indices of
         *                      the training examples that are included in the training set and the holdout set,
         *                      respectively
         * @param statistics    A reference to an object of type `IStatistics` that provides access to the statistics
         *                      which serve as a basis for learning rules
         * @return              An unique pointer to an object of type `IInstanceSampling` that has been created
         */
        virtual std::unique_ptr<IInstanceSampling> create(const BinaryCsrView& labelMatrix, BiPartition& partition,
                                                          IStatistics& statistics) const = 0;
};

/**
 * Defines an interface for all factories that allow to create instances of the type `IInstanceSampling` that can be
 * used in regression problems.
 */
class IRegressionInstanceSamplingFactory {
    public:

        virtual ~IRegressionInstanceSamplingFactory() {}

        /**
         * Creates and returns a new object of type `IInstanceSampling`.
         *
         * @param regressionMatrix  A reference to an object of type `CContiguousView` that provides access to the
         *                          regression scores of the training examples
         * @param partition         A reference to an object of type `SinglePartition` that provides access to the
         *                          indices of the training examples that are included in the training set
         * @param statistics        A reference to an object of type `IStatistics` that provides access to the
         *                          statistics which serve as a basis for learning rules
         * @return                  An unique pointer to an object of type `IInstanceSampling` that has been created
         */
        virtual std::unique_ptr<IInstanceSampling> create(const CContiguousView<const float32>& regressionMatrix,
                                                          const SinglePartition& partition,
                                                          IStatistics& statistics) const = 0;

        /**
         * Creates and returns a new object of type `IInstanceSampling`.
         *
         * @param regressionMatrix  A reference to an object of type `CContiguousView` that provides access to the
         *                          regression scores of the training examples
         * @param partition         A reference to an object of type `BiPartition` that provides access to the indices
         *                          of the training examples that are included in the training set and the holdout set,
         *                          respectively
         * @param statistics        A reference to an object of type `IStatistics` that provides access to the
         *                          statistics which serve as a basis for learning rules
         * @return                  An unique pointer to an object of type `IInstanceSampling` that has been created
         */
        virtual std::unique_ptr<IInstanceSampling> create(const CContiguousView<const float32>& regressionMatrix,
                                                          BiPartition& partition, IStatistics& statistics) const = 0;

        /**
         * Creates and returns a new object of type `IInstanceSampling`.
         *
         * @param regressionMatrix  A reference to an object of type `CsrView` that provides access to the regression
         *                          scores of the training examples
         * @param partition         A reference to an object of type `SinglePartition` that provides access to the
         *                          indices of the training examples that are included in the training set
         * @param statistics        A reference to an object of type `IStatistics` that provides access to the
         *                          statistics which serve as a basis for learning rules
         * @return                  An unique pointer to an object of type `IInstanceSampling` that has been created
         */
        virtual std::unique_ptr<IInstanceSampling> create(const CsrView<const float32>& regressionMatrix,
                                                          const SinglePartition& partition,
                                                          IStatistics& statistics) const = 0;

        /**
         * Creates and returns a new object of type `IInstanceSampling`.
         *
         * @param regressionMatrix  A reference to an object of type `CsrView` that provides access to the regression
         *                          scores of the training examples
         * @param partition         A reference to an object of type `BiPartition` that provides access to the indices
         *                          of the training examples that are included in the training set and the holdout set,
         *                          respectively
         * @param statistics        A reference to an object of type `IStatistics` that provides access to the
         *                          statistics which serve as a basis for learning rules
         * @return                  An unique pointer to an object of type `IInstanceSampling` that has been created
         */
        virtual std::unique_ptr<IInstanceSampling> create(const CsrView<const float32>& regressionMatrix,
                                                          BiPartition& partition, IStatistics& statistics) const = 0;
};

/**
 * Defines an interface for all classes that allow to configure a method for sampling instances that can be used in
 * classification problems.
 */
class IClassificationInstanceSamplingConfig {
    public:

        virtual ~IClassificationInstanceSamplingConfig() {}

        /**
         * Creates and returns a new object of type `IClassificationInstanceSamplingFactory` according to the specified
         * configuration.
         *
         * @return An unique pointer to an object of type `IClassificationInstanceSamplingFactory` that has been created
         */
        virtual std::unique_ptr<IClassificationInstanceSamplingFactory> createClassificationInstanceSamplingFactory()
          const = 0;
};

/**
 * Defines an interface for all classes that allow to configure a method for sampling instances that can be used in
 * regression problems.
 */
class IRegressionInstanceSamplingConfig {
    public:

        virtual ~IRegressionInstanceSamplingConfig() {}

        /**
         * Creates and returns a new object of type `IRegressionInstanceSamplingFactory` according to the specified
         * configuration.
         *
         * @return An unique pointer to an object of type `IRegressionInstanceSamplingFactory` that has been created
         */
        virtual std::unique_ptr<IRegressionInstanceSamplingFactory> createRegressionInstanceSamplingFactory() const = 0;
};
