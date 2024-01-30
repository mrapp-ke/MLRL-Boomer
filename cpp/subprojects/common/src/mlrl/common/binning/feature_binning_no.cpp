#include "mlrl/common/binning/feature_binning_no.hpp"

#include "mlrl/common/thresholds/thresholds_exact.hpp"

/**
 * Allows to create instances of the type `IFeatureBinning` that do not actualy perform any feature binning.
 */
class NoFeatureBinningFactory final : public IFeatureBinningFactory {
    public:

        std::unique_ptr<IFeatureBinning> create() const override {
            return nullptr;
        }
};

NoFeatureBinningConfig::NoFeatureBinningConfig(const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
    : multiThreadingConfigPtr_(multiThreadingConfigPtr) {}

std::unique_ptr<IThresholdsFactory> NoFeatureBinningConfig::createThresholdsFactory(
  const IFeatureMatrix& featureMatrix, const ILabelMatrix& labelMatrix) const {
    uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, labelMatrix.getNumLabels());
    return std::make_unique<ExactThresholdsFactory>(std::make_unique<NoFeatureBinningFactory>(), numThreads);
}
