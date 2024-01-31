/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/binning/feature_binning.hpp"

namespace boosting {

    /**
     * Allows to configure a method that automatically decides whether feature binning should be used or not.
     */
    class AutomaticFeatureBinningConfig final : public IFeatureBinningConfig {
        public:

            /**
             * @see `IFeatureBinningConfig::createFeatureBinningFactory`
             */
            std::unique_ptr<IFeatureBinningFactory> createFeatureBinningFactory(
              const IFeatureMatrix& featureMatrix, const ILabelMatrix& labelMatrix) const override;
    };

}
