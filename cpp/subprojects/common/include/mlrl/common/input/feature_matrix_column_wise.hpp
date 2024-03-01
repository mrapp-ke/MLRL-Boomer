/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/feature_matrix.hpp"
#include "mlrl/common/input/feature_type.hpp"
#include "mlrl/common/input/feature_vector.hpp"

#include <memory>

/**
 * Defines an interface for all feature matrices that provide column-wise access to the feature values of examples.
 */
class MLRLCOMMON_API IColumnWiseFeatureMatrix : public IFeatureMatrix {
    public:

        virtual ~IColumnWiseFeatureMatrix() override {}

        /**
         * Creates and returns a feature vector that stores the feature values of the available examples for a certain
         * feature.
         *
         * @param featureIndex  The index of the feature
         * @param featureType   A reference to an object of type `IFeatureType` that represents the type of the feature
         * @return              An unique pointer to an object of type `IFeatureVector` that has been created
         */
        virtual std::unique_ptr<IFeatureVector> createFeatureVector(uint32 featureIndex,
                                                                    const IFeatureType& featureType) const = 0;
};
