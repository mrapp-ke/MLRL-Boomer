/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"
#include "common/macros.hpp"


/**
 * Defines an interface for all classes that provide information about the types of individual features.
 */
class MLRLCOMMON_API IFeatureInfo {

    public:

        /**
         * An enum that specifies all supported types of features.
         */
        enum FeatureType : uint8 {
            BINARY = 0,
            NOMINAL = 1,
            NUMERICAL_OR_ORDINAL = 2
        };

        virtual ~IFeatureInfo() { };

        /**
         * Returns the type of the feature at a specific index.
         *
         * @return  A value of the enum `FeatureType` that specifies the type of the feature at the given index
         */
        virtual FeatureType getFeatureType(uint32 featureIndex) const = 0;

};
