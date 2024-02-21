/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"

#include <memory>

// Forward declarations
class IFeatureSpace;
class IFeatureSubspace;

/**
 * Defines an interface for one-dimensional vectors that provide access to weights.
 */
class IWeightVector {
    public:

        virtual ~IWeightVector() {}

        /**
         * Returns whether the vector contains any zero weights or not.
         *
         * @return True, if the vector contains any zero weights, false otherwise
         */
        virtual bool hasZeroWeights() const = 0;

        /**
         * Creates and returns a new instance of type `IFeatureSubspace` that uses the weights in this vector for the
         * training examples it includes.
         *
         * @param featureSpace  A reference to an object of type `IFeatureSpace` that should be used to create the
         *                      instance
         * @return              An unique pointer to an object of type `IFeatureSubspace` that has been created
         */
        virtual std::unique_ptr<IFeatureSubspace> createFeatureSubspace(IFeatureSpace& featureSpace) const = 0;
};
