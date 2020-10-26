/**
 * Provides interfaces and classes that provide access to indices that allow to restrict the access to data that is
 * stored in matrices or vectors to a subset of the available data.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "data.h"
#include <memory>

// Forward declarations
class AbstractThresholds;
class IThresholdsSubset;
class IWeightVector;
class AbstractStatistics;
class IStatisticsSubset;


/**
 * Defines an interface for all one-dimensional vectors that provide random access to indices.
 */
class IIndexVector : virtual public IRandomAccessVector<uint32> {

    public:

        virtual ~IIndexVector() { };

        /**
         * Creates and returns a new subset of the given thresholds that only contains the labels whose indices are
         * stored in this vector.
         *
         * @param thresholds    A reference to an object of type `AbstractThresholds` that should be used to create the
         *                      subset
         * @param weights       A reference to an object of type `IWeightVector` that provides access to the weights of
         *                      individual training examples
         * @return              An unique pointer to an object of type `IThresholdsSubset` that has been created
         */
        virtual std::unique_ptr<IThresholdsSubset> createSubset(AbstractThresholds& thresholds,
                                                                IWeightVector& weights) const = 0;

        /**
         * Creates and returns a new subset of the given statistics that only contains the labels whose indices are
         * stored in this vector.
         *
         * @param statistics    A reference to an object of type `AbstractStatistics` that should be used to create the
         *                      subset
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        // TODO Remove arguments `numLabelIndices` and `labelIndices`
        virtual std::unique_ptr<IStatisticsSubset> createSubset(const AbstractStatistics& statistics,
                                                                uint32 numLabelIndices,
                                                                const uint32* labelIndices) const = 0;

};

/**
 * An one-dimensional vector that provides random access to a fixed number of indices stored in a C-contiguous array.
 */
class DenseIndexVector : public DenseVector<uint32>, virtual public IIndexVector {

    public:

        /**
         * @param numElements The number of elements in the vector
         */
        DenseIndexVector(uint32 numElements);

        std::unique_ptr<IThresholdsSubset> createSubset(AbstractThresholds& thresholds,
                                                        IWeightVector& weights) const override;

        std::unique_ptr<IStatisticsSubset> createSubset(const AbstractStatistics& statistics, uint32 numLabelIndices,
                                                        const uint32* labelIndices) const override;

};

/**
 * An one-dimensional vector that provides random access to all indices within a continuous range [0, numIndices).
 */
class RangeIndexVector : virtual public IIndexVector {

    private:

        uint32 numElements_;

    public:

        /**
         * @param numElements The number of indices, the vector provides access to
         */
        RangeIndexVector(uint32 numElements);

        uint32 getNumElements() const override;

        uint32 getValue(uint32 pos) const override;

        std::unique_ptr<IThresholdsSubset> createSubset(AbstractThresholds& thresholds,
                                                        IWeightVector& weights) const override;

        std::unique_ptr<IStatisticsSubset> createSubset(const AbstractStatistics& statistics, uint32 numLabelIndices,
                                                        const uint32* labelIndices) const override;

};
