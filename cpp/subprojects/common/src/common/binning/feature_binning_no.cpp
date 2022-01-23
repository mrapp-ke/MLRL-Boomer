#include "common/binning/feature_binning_no.hpp"
#include "common/thresholds/thresholds_exact.hpp"


NoFeatureBinningConfig::NoFeatureBinningConfig(const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
    : multiThreadingConfigPtr_(multiThreadingConfigPtr) {

}

std::unique_ptr<IThresholdsFactory> NoFeatureBinningConfig::configure(const IFeatureMatrix& featureMatrix,
                                                                      const ILabelMatrix& labelMatrix) const {
    uint32 numThreads = multiThreadingConfigPtr_->configure(featureMatrix, labelMatrix);
    return std::make_unique<ExactThresholdsFactory>(numThreads);
}
