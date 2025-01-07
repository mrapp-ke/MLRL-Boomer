/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/iterator/iterator_equal.hpp"
#include "mlrl/common/sampling/weight_vector.hpp"

#include <memory>

/**
 * An one-dimensional vector that provides random access to a fixed number of equal weights.
 */
class EqualWeightVector final : public IWeightVector {
    private:

        const uint32 numElements_;

    public:

        /**
         * @param numElements The number of elements in the vector
         */
        EqualWeightVector(uint32 numElements);

        /**
         * The type of the weights, the vector provides access to.
         */
        typedef uint32 weight_type;

        /**
         * An iterator that provides read-only access to the elements in the vector.
         */
        typedef EqualIterator<weight_type> const_iterator;

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
         * @return The number of elements
         */
        uint32 getNumElements() const;

        /**
         * Returns the number of non-zero weights.
         *
         * @return The number of non-zero weights
         */
        uint32 getNumNonZeroWeights() const;

        /**
         * Returns the weight at a specific position.
         *
         * @param pos   The position
         * @return      The weight at the specified position
         */
        weight_type operator[](uint32 pos) const;

        bool hasZeroWeights() const override;

        std::unique_ptr<IFeatureSubspace> createFeatureSubspace(IFeatureSpace& featureSpace) const override;
};
