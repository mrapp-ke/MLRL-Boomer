/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/vector_sparse_array.hpp"
#include "mlrl/common/input/missing_feature_vector.hpp"
#include "mlrl/common/thresholds/coverage_mask.hpp"

/**
 * Defines an interface for all one-dimensional vectors that store the values of training examples for a certain
 * feature.
 */
class IFeatureVector : public IOneDimensionalView {
    public:

        virtual ~IFeatureVector() override {};

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
class FeatureVector final : public MissingFeatureVector {
    private:

        SparseArrayVector<float32> vector_;

    public:

        /**
         * @param numElements The number of elements in the vector
         */
        FeatureVector(uint32 numElements);

        /**
         * An iterator that provides access to the feature values in the vector and allows to modify them.
         */
        typedef SparseArrayVector<float32>::iterator iterator;

        /**
         * An iterator that provides read-only access to the feature values in the vector.
         */
        typedef SparseArrayVector<float32>::const_iterator const_iterator;

        /**
         * Returns an `iterator` to the beginning of the vector.
         *
         * @return An `iterator` to the beginning
         */
        iterator begin();

        /**
         * Returns an `iterator` to the end of the vector.
         *
         * @return An `iterator` to the end
         */
        iterator end();

        /**
         * Returns a `const_iterator` to the beginning of the vector.
         *
         * @return A `const_iterator` to the beginning
         */
        const_iterator cbegin() const;

        /**
         * Returns a `const_iterator` to the end of the vector.
         *
         * @return A `const_iterator` to the end
         */
        const_iterator cend() const;

        /**
         * Returns the number of elements in the vector.
         *
         * @return The number of elements in the vector
         */
        uint32 getNumElements() const;

        /**
         * Sets the number of elements in the vector.
         *
         * @param numElements   The number of elements to be set
         * @param freeMemory    True, if unused memory should be freed, if possible, false otherwise
         */
        void setNumElements(uint32 numElements, bool freeMemory);

        /**
         * Sorts the elements in the vector in ascending order based on their values.
         */
        void sortByValues();
};
