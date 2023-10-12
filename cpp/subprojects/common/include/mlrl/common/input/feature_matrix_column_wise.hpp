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
class MLRLCOMMON_API IColumnWiseFeatureMatrix : virtual public IFeatureMatrix {
    public:

        virtual ~IColumnWiseFeatureMatrix() override {};

        /**
         * Fetches a feature vector that stores the indices of the training examples, as well as their feature values,
         * for a specific feature and stores it in a given unique pointer.
         *
         * @param featureIndex      The index of the feature
         * @param featureVectorPtr  An unique pointer to an object of type `FeatureVector` that should be used to store
         *                          the feature vector
         */
        // TODO Remove
        virtual void fetchFeatureVector(uint32 featureIndex,
                                        std::unique_ptr<FeatureVector>& featureVectorPtr) const = 0;

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
