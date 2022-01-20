/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/binning/feature_binning.hpp"


/**
 * Allows to configure a method that automatically whether feature binning should be used or not.
 */
class AutomaticFeatureBinningConfig final : public IFeatureBinningConfig {

    public:

        std::unique_ptr<IThresholdsFactory> create(const IFeatureMatrix& featureMatrix) const override;

};
