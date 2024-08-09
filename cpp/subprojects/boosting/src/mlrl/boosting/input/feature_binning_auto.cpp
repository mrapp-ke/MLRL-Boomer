#include "mlrl/boosting/input/feature_binning_auto.hpp"

#include "mlrl/common/input/feature_binning_equal_width.hpp"
#include "mlrl/common/input/feature_binning_no.hpp"

namespace boosting {

    std::unique_ptr<IFeatureBinningFactory> AutomaticFeatureBinningConfig::createFeatureBinningFactory(
      const IFeatureMatrix& featureMatrix, const IOutputMatrix& outputMatrix) const {
        if (!featureMatrix.isSparse() && featureMatrix.getNumExamples() > 200000) {
            return EqualWidthFeatureBinningConfig().createFeatureBinningFactory(featureMatrix, outputMatrix);
        } else {
            return NoFeatureBinningConfig().createFeatureBinningFactory(featureMatrix, outputMatrix);
        }
    }

}
