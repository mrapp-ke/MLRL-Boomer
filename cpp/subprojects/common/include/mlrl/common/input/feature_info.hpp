/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"
#include "mlrl/common/dll_exports.hpp"
#include "mlrl/common/input/feature_type.hpp"

#include <memory>

/**
 * Defines an interface for all classes that provide information about the types of individual features.
 */
class MLRLCOMMON_API IFeatureInfo {
    public:

        virtual ~IFeatureInfo() {};

        /**
         * Creates and returns a new object of type `IFeatureType` that corresponds to the type of the feature at a
         * specific index.
         *
         * @return  An unique pointer to an object of the type `IFeatureType` that has been created
         */
        virtual std::unique_ptr<IFeatureType> createFeatureType(uint32 featureIndex) const = 0;
};
