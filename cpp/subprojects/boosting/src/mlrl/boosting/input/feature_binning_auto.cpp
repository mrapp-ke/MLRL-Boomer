#include "mlrl/boosting/input/feature_binning_auto.hpp"

#include "mlrl/common/input/feature_binning_equal_width.hpp"
#include "mlrl/common/input/feature_binning_no.hpp"

namespace boosting {

    std::unique_ptr<IFeatureBinningFactory> AutomaticFeatureBinningConfig::createFeatureBinningFactory(
      const IFeatureMatrix& featureMatrix, const ILabelMatrix& labelMatrix) const {
        if (!featureMatrix.isSparse() && featureMatrix.getNumExamples() > 200000) {
            return EqualWidthFeatureBinningConfig().createFeatureBinningFactory(featureMatrix, labelMatrix);
        } else {
            return NoFeatureBinningConfig().createFeatureBinningFactory(featureMatrix, labelMatrix);
        }
    }

}
