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

        /**
         * Marks all features as numerical/ordinal.
         */
        virtual void setAllNumerical() = 0;

        /**
         * Marks all features as binary.
         */
        virtual void setAllBinary() = 0;

        /**
         * Marks all features as nominal.
         */
        virtual void setAllNominal() = 0;

};

/**
 * Creates and returns a new object of type `IEqualFeatureInfo`.
 *
 * @return An unique pointer to an object of type `IEqualFeatureInfo` that has been created
 */
MLRLCOMMON_API std::unique_ptr<IEqualFeatureInfo> createEqualFeatureInfo();
