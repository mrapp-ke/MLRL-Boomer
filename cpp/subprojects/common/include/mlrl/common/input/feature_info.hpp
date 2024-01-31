/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/binning/feature_binning.hpp"
#include "mlrl/common/data/types.hpp"
#include "mlrl/common/input/feature_type.hpp"
#include "mlrl/common/util/dll_exports.hpp"

#include <memory>

/**
 * Defines an interface for all classes that provide information about the types of individual features.
 */
class MLRLCOMMON_API IFeatureInfo {
    public:

        virtual ~IFeatureInfo() {}

        /**
         * Creates and returns a new object of type `IFeatureType` that corresponds to the type of the feature at a
         * specific index.
         *
         * @param featureIndex          The index of the feature
         * @param featureBinningFactory A reference to an object of type `IFeatureBinningFactory` that allows to create
         *                              implementations of the binning method to be used for assigning numerical feature
         *                              values to bins
         * @return                      An unique pointer to an object of the type `IFeatureType` that has been created
         */
        virtual std::unique_ptr<IFeatureType> createFeatureType(
          uint32 featureIndex, const IFeatureBinningFactory& featureBinningFactory) const = 0;
};
