/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/vector_sparse_array.hpp"
#include "mlrl/common/input/feature_vector_common.hpp"

/**
 * A feature vector that stores the values of training examples for a certain numerical feature, except for the examples
 * associated with a sparse value.
 */
class NumericalFeatureVector final : public AbstractFeatureVector {
    private:

        SparseArrayVector<float32> vector_;

        const float32 sparseValue_;

    public:

        /**
         * @param numElements   The number of elements in the vector, excluding those associated with the sparse value
         * @param sparseValue   The value of sparse elements not explicitly stored in the vector
         */
        NumericalFeatureVector(uint32 numElements, float32 sparseValue);

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
         * Returns the value of sparse elements not explicitly stored in the vector.
         *
         * @return The value of sparse elements
         */
        float32 getSparseValue() const;

        /**
         * Sorts the elements in the vector in ascending order based on their values.
         */
        void sortByValues();

        /**
         * Sets the number of elements in the vector.
         *
         * @param numElements   The number of elements to be set
         * @param freeMemory    True, if unused memory should be freed, if possible, false otherwise
         */
        void setNumElements(uint32 numElements, bool freeMemory);

        uint32 getNumElements() const override;
};
