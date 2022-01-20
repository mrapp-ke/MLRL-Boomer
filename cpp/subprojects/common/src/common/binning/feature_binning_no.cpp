#include "common/binning/feature_binning_no.hpp"
#include "common/thresholds/thresholds_exact.hpp"


std::unique_ptr<IThresholdsFactory> NoFeatureBinningConfig::create(const IFeatureMatrix& featureMatrix) const {
    uint32 numThreads = 1; // TODO use correct value
    return std::make_unique<ExactThresholdsFactory>(numThreads);
}
