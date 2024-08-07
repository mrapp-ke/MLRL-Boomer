/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/feature_binning.hpp"

#include <memory>

/**
 * Allows to configure a method that does not actually perform any feature binning.
 */
class NoFeatureBinningConfig final : public IFeatureBinningConfig {
    public:

        std::unique_ptr<IFeatureBinningFactory> createFeatureBinningFactory(
          const IFeatureMatrix& featureMatrix, const IOutputMatrix& outputMatrix) const override;
};
