/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/indexed_value.hpp"
#include "mlrl/common/data/vector_dense.hpp"
#include "mlrl/common/input/interval.hpp"
#include "mlrl/common/input/missing_feature_vector.hpp"
#include "mlrl/common/statistics/statistics_weighted.hpp"
#include "mlrl/common/thresholds/coverage_mask.hpp"

/**
 * Defines an interface for all one-dimensional vectors that store the values of training examples for a certain
 * feature.
 */
class IFeatureVector {
    public:

        virtual ~IFeatureVector() {}

        /**
         * Updates a given `CoverageMask` and `IWeightedStatistics` depending on the indices of training examples
         * included in a specific `Interval`.
         *
         * @param interval      A reference to an object of type `Interval` that includes the indices of all covered
         *                      training examples
         * @param coverageMask  A reference to an object of type `CoverageMask` that should be updated
         * @param statistics    A reference to an object of type `IWeightedStatistics` that should be updated
         */
        virtual void updateCoverageMaskAndStatistics(const Interval& interval, CoverageMask& coverageMask,
                                                     IWeightedStatistics& statistics) const = 0;

        /**
         * Creates and returns a copy of this vector that does only store the feature values of training examples
         * included in a given `Interval`.
         *
         * @param existing  A reference to an unique pointer that stores an object of type `IFeatureVector` that may be
         *                  reused or a null pointer, if no such object is available
         * @param interval  A reference to an object of type `Interval` that includes the indices of the training
         *                  examples to be retained
         * @return          An unique pointer to an object of type `IFeatureVector` that has been created
         */
        virtual std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                            const Interval& interval) const = 0;

        /**
         * Creates and returns a copy of this vector that does only store the feature values of training examples marked
         * as covered according to a given `CoverageMask`.
         *
         * @param existing      A reference to an unique pointer that stores an object of type `IFeatureVector` that may
         *                      be reused or a null pointer, if no such object is available
         * @param coverageMask  A reference to an object of type `CoverageMask` that specifies the indices of the
         *                      training examples to be retained
         * @return              An unique pointer to an object of type `IFeatureVector` that has been created
         */
        virtual std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                            const CoverageMask& coverageMask) const = 0;
};

/**
 * An one-dimensional sparse vector that stores the values of training examples for a certain feature, as well as the
 * indices of examples with missing feature values.
 */
// TODO Remove
class FeatureVector final
    : public ResizableVectorDecorator<DenseVectorDecorator<ResizableVector<IndexedValue<float32>>>>,
      public OldMissingFeatureVector {
    public:

        /**
         * @param numElements The number of elements in the vector
         */
        FeatureVector(uint32 numElements);

        /**
         * Sorts the elements in the vector in ascending order based on their values.
         */
        void sortByValues();
};
