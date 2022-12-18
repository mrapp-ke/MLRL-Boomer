#include "common/input/feature_info_equal.hpp"


/**
 * An implementation of the type `IEqualFeatureInfo` that stores the type of all features.
 */
class EqualFeatureInfo final : public IEqualFeatureInfo {

    private:

        FeatureType featureType_;

    public:

        /**
         * @param featureType A value of the enum `FeatureType` that specifies the type of all features
         */
        EqualFeatureInfo(FeatureType featureType)
            : featureType_(featureType) {

        }

        FeatureType getFeatureType(uint32 featureIndex) const override {
            return featureType_;
        }

};

std::unique_ptr<IEqualFeatureInfo> createEqualFeatureInfo(IFeatureInfo::FeatureType featureType) {
    return std::make_unique<EqualFeatureInfo>(featureType);
}
