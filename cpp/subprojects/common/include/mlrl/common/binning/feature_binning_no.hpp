/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/binning/feature_binning.hpp"

/**
 * Allows to configure a method that does not actually perform any feature binning.
 */
class NoFeatureBinningConfig final : public IFeatureBinningConfig {
    public:

        std::unique_ptr<IFeatureBinningFactory> createFeatureBinningFactory(
          const IFeatureMatrix& featureMatrix, const ILabelMatrix& labelMatrix) const override;
};
