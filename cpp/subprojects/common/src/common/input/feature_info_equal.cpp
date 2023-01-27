#include "common/input/feature_info_equal.hpp"
#include "common/input/feature_type_binary.hpp"
#include "common/input/feature_type_nominal.hpp"
#include "common/input/feature_type_numerical.hpp"

/**
 * An implementation of the type `IEqualFeatureInfo` that stores the type of all features.
 *
 * @tparam FeatureType The type of all features
 */
template<typename FeatureType>
class EqualFeatureInfo final : public IEqualFeatureInfo {
    public:

        std::unique_ptr<IFeatureType> createFeatureType(uint32 featureIndex) const override {
            return std::make_unique<FeatureType>();
        }
};

std::unique_ptr<IEqualFeatureInfo> createBinaryFeatureInfo() {
    return std::make_unique<EqualFeatureInfo<BinaryFeatureType>>();
}

std::unique_ptr<IEqualFeatureInfo> createNominalFeatureInfo() {
    return std::make_unique<EqualFeatureInfo<NominalFeatureType>>();
}

std::unique_ptr<IEqualFeatureInfo> createNumericalFeatureInfo() {
    return std::make_unique<EqualFeatureInfo<NumericalFeatureType>>();
}
