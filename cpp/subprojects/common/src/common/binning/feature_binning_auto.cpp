#include "common/binning/feature_binning_auto.hpp"
#include "common/binning/feature_binning_equal_width.hpp"
#include "common/binning/feature_binning_no.hpp"


std::unique_ptr<IThresholdsFactory> AutomaticFeatureBinningConfig::configure(
        const IFeatureMatrix& featureMatrix) const {
    if (!featureMatrix.isSparse() && featureMatrix.getNumRows() > 200000) {
        return EqualWidthFeatureBinningConfig().configure(featureMatrix);
    } else {
        return NoFeatureBinningConfig().configure(featureMatrix);
    }
}
