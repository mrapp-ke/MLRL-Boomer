/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/input/label_vector.hpp"
#include <memory>

// Forward declarations
class IStatisticsProvider;
class IStatisticsProviderFactory;
class IInstanceSubSampling;
class IInstanceSubSamplingFactory;
class SinglePartition;
class BiPartition;


/**
 * Defines an interface for all label matrices that provide access to the labels of the training examples.
 */
class ILabelMatrix {

    public:

        virtual ~ILabelMatrix() { };

        /**
         * Returns the number of available examples.
         *
         * @return The number of examples
         */
        virtual uint32 getNumRows() const = 0;

        /**
         * Returns the number of available labels.
         *
         * @return The number of labels
         */
        virtual uint32 getNumCols() const = 0;

        /**
         * Creates and returns a label vector that corresponds to a specific row in the label matrix.
         *
         * @param row   The row
         * @return      An unique pointer to an object of type `LabelVector` that has been created
         */
        virtual std::unique_ptr<LabelVector> getLabelVector(uint32 row) const = 0;

        /**
         * Creates and returns a new instance of the class `IStatisticsProvider`, based on the type of this label
         * matrix.
         *
         * @param factory   A reference to an object of type `IStatisticsProviderFactory` that should be used to create
         *                  the instance
         * @return          An unique pointer to an object of type `IStatisticsProvider` that has been created
         */
        virtual std::unique_ptr<IStatisticsProvider> createStatisticsProvider(
            const IStatisticsProviderFactory& factory) const = 0;

        /**
         * Creates and returns a new instance of the class `IInstanceSubSampling`, based on the type of this label
         * matrix.
         *
         * @param factory   A reference to an object of type `IInstanceSubSamplingFactory` that should be used to create
         *                  the instance
         * @param partition A reference to an object of type `SinglePartition` that provides access to the indices of
         *                  the training examples that are included in the training set
         * @return          An unique pointer to an object of type `IInstanceSubSampling` that has been created
         */
        virtual std::unique_ptr<IInstanceSubSampling> createInstanceSubSampling(
            const IInstanceSubSamplingFactory& factory, const SinglePartition& partition) const = 0;

        /**
         * Creates and returns a new instance of the class `IInstanceSubSampling`, based on the type of this label
         * matrix.
         *
         * @param factory   A reference to an object of type `IInstanceSubSamplingFactory` that should be used to create
         *                  the instance
         * @param partition A reference to an object of type `BiPartition` that provides access to the indices of the
         *                  training examples that are included in the training set and the holdout set, respectively
         * @return          An unique pointer to an object of type `IInstanceSubSampling` that has been created
         */
        virtual std::unique_ptr<IInstanceSubSampling> createInstanceSubSampling(
            const IInstanceSubSamplingFactory& factory, BiPartition& partition) const = 0;

};
