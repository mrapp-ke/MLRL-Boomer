/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"
#include "common/input/feature_type.hpp"
#include "common/macros.hpp"
#include <memory>


/**
 * Defines an interface for all classes that provide information about the types of individual features.
 */
class MLRLCOMMON_API IFeatureInfo {

    public:

        virtual ~IFeatureInfo() { };

        /**
         * Returns the type of the feature at a specific index.
         *
         * @return  An unique pointer to an object of the type `IFeatureType` that represents the type of the feature
         */
        virtual std::unique_ptr<IFeatureType> getFeatureType(uint32 featureIndex) const = 0;

};
