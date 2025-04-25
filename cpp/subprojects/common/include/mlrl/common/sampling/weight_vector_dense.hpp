/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/vector_dense.hpp"
#include "mlrl/common/sampling/weight_vector.hpp"

#include <memory>

/**
 * An one-dimensional vector that provides random access to a fixed number of weights stored in a C-contiguous array.
 *
 * @tparam T The type of the weights
 */
template<typename T>
class DenseWeightVector final : public ClearableViewDecorator<DenseVectorDecorator<AllocatedVector<T>>>,
                                public IWeightVector {
    private:

        uint32 numNonZeroWeights_;

    public:

        /**
         * @param numElements   The number of elements in the vector
         * @param init          True, if all elements in the vector should be value-initialized, false otherwise
         */
        DenseWeightVector(uint32 numElements, bool init = false);

        /**
         * The type of the weights, the vector provides access to.
         */
        typedef T weight_type;

        /**
         * Sets the weight at a specific position.
         *
         * @param pos       The position
         * @param weight    The weight to be set
         */
        void set(uint32 pos, weight_type weight);

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
