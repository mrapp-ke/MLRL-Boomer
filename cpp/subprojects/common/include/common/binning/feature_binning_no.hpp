/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/binning/feature_binning.hpp"


/**
 * Allows to configure a method that does not actually perform any feature binning.
 */
class NoFeatureBinningConfig final : public IFeatureBinningConfig {

    public:

        std::unique_ptr<IThresholdsFactory> create(const IFeatureMatrix& featureMatrix) const override;

};
