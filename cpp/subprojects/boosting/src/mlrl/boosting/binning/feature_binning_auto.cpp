#include "mlrl/boosting/binning/feature_binning_auto.hpp"

#include "mlrl/common/binning/feature_binning_equal_width.hpp"
#include "mlrl/common/binning/feature_binning_no.hpp"

namespace boosting {

    AutomaticFeatureBinningConfig::AutomaticFeatureBinningConfig(
      const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
        : multiThreadingConfigPtr_(multiThreadingConfigPtr) {}

    std::unique_ptr<IThresholdsFactory> AutomaticFeatureBinningConfig::createThresholdsFactory(
      const IFeatureMatrix& featureMatrix, const ILabelMatrix& labelMatrix) const {
        if (!featureMatrix.isSparse() && featureMatrix.getNumExamples() > 200000) {
            return EqualWidthFeatureBinningConfig(multiThreadingConfigPtr_)
              .createThresholdsFactory(featureMatrix, labelMatrix);
        } else {
            return NoFeatureBinningConfig(multiThreadingConfigPtr_).createThresholdsFactory(featureMatrix, labelMatrix);
        }
    }

}
