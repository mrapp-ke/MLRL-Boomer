/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/vector_bit.hpp"
#include "mlrl/common/sampling/weight_vector.hpp"

#include <memory>

/**
 * An one-dimensional vector that provides random access to a fixed number of binary weights stored in a `BitVector`.
 */
class BitWeightVector final : public IWeightVector {
    private:

        BitVector vector_;

        uint32 numNonZeroWeights_;

    public:

        /**
         * @param numElements   The number of elements in the vector
         * @param init          True, if all elements in the vector should be value-initialized, false otherwise
         */
        BitWeightVector(uint32 numElements, bool init = false);

        /**
         * The type of the weights, the vector provides access to.
         */
        typedef bool weight_type;

        /**
         * Returns the number of elements in the vector.
         *
         * @return The number of elements
         */
        uint32 getNumElements() const;

        /**
         * Returns the weight at a specific position.
         *
         * @param pos   The position
         * @return      The weight at the specified position
         */
        weight_type operator[](uint32 pos) const;

        /**
         * Sets the weight at a specific position.
         *
         * @param pos       The position
         * @param weight    The weight to be set
         */
        void set(uint32 pos, bool weight);

        /**
         * Sets all weights to zero.
         */
        void clear();

        /**
         * Returns the number of non-zero weights.
         *
         * @return The number of non-zero weights
         */
        uint32 getNumNonZeroWeights() const;

        /**
         * Sets the number of non-zero weights.
         *
         * @param numNonZeroWeights The number of non-zero weights to be set
         */
        void setNumNonZeroWeights(uint32 numNonZeroWeights);

        bool hasZeroWeights() const override;

        std::unique_ptr<IFeatureSubspace> createFeatureSubspace(IFeatureSpace& featureSpace) const override;
};
