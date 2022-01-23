#include "boosting/binning/feature_binning_auto.hpp"
#include "common/binning/feature_binning_equal_width.hpp"
#include "common/binning/feature_binning_no.hpp"


namespace boosting {

    AutomaticFeatureBinningConfig::AutomaticFeatureBinningConfig(
            const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
        : multiThreadingConfigPtr_(multiThreadingConfigPtr) {

    }

    std::unique_ptr<IThresholdsFactory> AutomaticFeatureBinningConfig::configure(
            const IFeatureMatrix& featureMatrix, const ILabelMatrix& labelMatrix) const {
        if (!featureMatrix.isSparse() && featureMatrix.getNumRows() > 200000) {
            return EqualWidthFeatureBinningConfig(multiThreadingConfigPtr_).configure(featureMatrix, labelMatrix);
        } else {
            return NoFeatureBinningConfig(multiThreadingConfigPtr_).configure(featureMatrix, labelMatrix);
        }
    }

}
