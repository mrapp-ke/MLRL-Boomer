/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/input/feature_info.hpp"
#include "common/macros.hpp"
#include <memory>


/**
 * Defines an interface for all classes that provide information about the types of individual features in cases where
 * all features are of the same type, i.e., where all features are either binary, nominal or numerical/ordinal.
 */
class MLRLCOMMON_API IEqualFeatureInfo : public IFeatureInfo {

    public:

        virtual ~IEqualFeatureInfo() override { };

};

/**
 * Creates and returns a new object of type `IEqualFeatureInfo`.
 *
 * @param nominal   A value of the enum `IFeatureInfo::FeatureType` that specifies the type of all features
 * @return          An unique pointer to an object of type `IEqualFeatureInfo` that has been created
 */
MLRLCOMMON_API std::unique_ptr<IEqualFeatureInfo> createEqualFeatureInfo(IFeatureInfo::FeatureType featureType);
