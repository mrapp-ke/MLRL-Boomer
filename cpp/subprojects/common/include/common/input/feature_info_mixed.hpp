/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/input/feature_info.hpp"
#include <memory>


/**
 * Defines an interface for all classes that provide information about the types of individual features in cases where
 * different types of features, i.e., binary, nominal and numerical/ordinal ones, are available.
 */
class MLRLCOMMON_API IMixedFeatureInfo : public IFeatureInfo {

    public:

        virtual ~IMixedFeatureInfo() override { };

        /**
         * Sets the type of the feature at a specific index.
         *
         * @param featureIndex  The index of the feature
         * @param featureType   A value of the enum `FeatureType` that specifies the type of the feature
         */
        virtual void setFeatureType(uint32 featureIndex, FeatureType featureType) = 0;

};

/**
 * Creates and returns a new object of type `IMixedFeatureInfo`.
 *
 * @param numFeatures   The total number of available features
 * @return              An unique pointer to an object of type `IMixedFeatureInfo` that has been created
 */
MLRLCOMMON_API std::unique_ptr<IMixedFeatureInfo> createMixedFeatureInfo(uint32 numFeatures);
