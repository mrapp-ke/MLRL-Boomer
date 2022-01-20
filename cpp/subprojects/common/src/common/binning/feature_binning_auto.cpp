#include "common/binning/feature_binning_auto.hpp"
#include "common/binning/feature_binning_equal_width.hpp"
#include "common/binning/feature_binning_no.hpp"


std::unique_ptr<IThresholdsFactory> AutomaticFeatureBinningConfig::create(const IFeatureMatrix& featureMatrix) const {
    if (!featureMatrix.isSparse() && featureMatrix.getNumRows() > 200000) {
        return EqualWidthFeatureBinningConfig().create(featureMatrix);
    } else {
        return NoFeatureBinningConfig().create(featureMatrix);
    }
}
