/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/indexed_value.hpp"
#include "mlrl/common/data/vector_dense.hpp"
#include "mlrl/common/input/missing_feature_vector.hpp"
#include "mlrl/common/thresholds/coverage_mask.hpp"

/**
 * Defines an interface for all one-dimensional vectors that store the values of training examples for a certain
 * feature.
 */
class IFeatureVector : public IOneDimensionalView {
    public:

        virtual ~IFeatureVector() override {}

        /**
         * Creates and returns a copy of this vector that does only store the feature values between a given start and
         * end index.
         *
         * @param existing  A reference to an unique pointer that stores an object of type `IFeatureVector` that may be
         *                  reused or a null pointer, if no such object is available
         * @param start     The index of the first feature value to be retained (inclusive)
         * @param end       The index of the last feature value to be retained (exclusive)
         * @return          An unique pointer to an object of type `IFeatureVector` that has been created
         */
        virtual std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                            uint32 start, uint32 end) const = 0;

        /**
         * Creates and returns a copy of this vector that does only store the feature values of training examples marked
         * as covered according to a given `CoverageMask` are retained.
         *
         * @param existing      A reference to an unique pointer that stores an object of type `IFeatureVector` that may
         *                      be reused or a null pointer, if no such object is available
         * @param coverageMask  A reference to an object of type `CoverageMask` to be used
         * @return              An unique pointer to an object of type `IFeatureVector` that has been created
         */
        virtual std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                            const CoverageMask& coverageMask) const = 0;
};

/**
 * An one-dimensional sparse vector that stores the values of training examples for a certain feature, as well as the
 * indices of examples with missing feature values.
 */
class FeatureVector final
    : public ResizableVectorDecorator<DenseVectorDecorator<ResizableVector<IndexedValue<float32>>>>,
      public MissingFeatureVector {
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
