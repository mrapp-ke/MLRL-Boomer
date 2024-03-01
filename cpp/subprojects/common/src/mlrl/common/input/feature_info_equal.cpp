#include "mlrl/common/input/feature_info_equal.hpp"

#include "mlrl/common/input/feature_type_nominal.hpp"
#include "mlrl/common/input/feature_type_numerical.hpp"
#include "mlrl/common/input/feature_type_ordinal.hpp"

/**
 * An implementation of the type `IEqualFeatureInfo` that creates an object of type `OrdinalFeatureType` for each
 * feature.
 */
class OrdinalFeatureInfo final : public IEqualFeatureInfo {
    public:

        std::unique_ptr<IFeatureType> createFeatureType(
          uint32 featureIndex, const IFeatureBinningFactory& featureBinningFactory) const override {
            return std::make_unique<OrdinalFeatureType>();
        }
};

/**
 * An implementation of the type `IEqualFeatureInfo` that creates an object of type `NominalFeatureType` for each
 * feature.
 */
class NominalFeatureInfo final : public IEqualFeatureInfo {
    public:

        std::unique_ptr<IFeatureType> createFeatureType(
          uint32 featureIndex, const IFeatureBinningFactory& featureBinningFactory) const override {
            return std::make_unique<NominalFeatureType>();
        }
};

/**
 * An implementation of the type `IEqualFeatureInfo` that creates an object of type `IFeatureBinning` or
 * `NumericalFeatureType` for each feature, depending on whether feature binning should be used or not.
 */
class NumericalFeatureInfo final : public IEqualFeatureInfo {
    public:

        std::unique_ptr<IFeatureType> createFeatureType(
          uint32 featureIndex, const IFeatureBinningFactory& featureBinningFactory) const override {
            std::unique_ptr<IFeatureBinning> featureBinningPtr = featureBinningFactory.create();

            if (featureBinningPtr) {
                return featureBinningPtr;
            }

            return std::make_unique<NumericalFeatureType>();
        }
};

std::unique_ptr<IEqualFeatureInfo> createOrdinalFeatureInfo() {
    return std::make_unique<OrdinalFeatureInfo>();
}

std::unique_ptr<IEqualFeatureInfo> createNominalFeatureInfo() {
    return std::make_unique<NominalFeatureInfo>();
}

std::unique_ptr<IEqualFeatureInfo> createNumericalFeatureInfo() {
    return std::make_unique<NumericalFeatureInfo>();
}
