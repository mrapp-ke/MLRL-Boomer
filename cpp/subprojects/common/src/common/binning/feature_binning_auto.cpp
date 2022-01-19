#include "common/binning/feature_binning_auto.hpp"
#include "common/binning/feature_binning_no.hpp"


std::unique_ptr<IThresholdsFactory> AutomaticFeatureBinningConfig::create() const {
    // TODO Check if feature binning should be used
    return NoFeatureBinningConfig().create();
}
