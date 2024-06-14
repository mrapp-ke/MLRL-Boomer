#include "mlrl/common/input/feature_binning_no.hpp"

/**
 * Allows to create instances of the type `IFeatureBinning` that do not actualy perform any feature binning.
 */
class NoFeatureBinningFactory final : public IFeatureBinningFactory {
    public:

        std::unique_ptr<IFeatureBinning> create() const override {
            return nullptr;
        }
};

std::unique_ptr<IFeatureBinningFactory> NoFeatureBinningConfig::createFeatureBinningFactory(
  const IFeatureMatrix& featureMatrix, const IOutputMatrix& outputMatrix) const {
    return std::make_unique<NoFeatureBinningFactory>();
}
