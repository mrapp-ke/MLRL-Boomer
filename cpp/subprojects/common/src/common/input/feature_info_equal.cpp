#include "common/input/feature_info_equal.hpp"


/**
 * An implementation of the type `IEqualFeatureInfo` that stores the type of all features.
 */
class EqualFeatureInfo final : public IEqualFeatureInfo {

    private:

        FeatureType featureType_;

    public:

        EqualFeatureInfo()
            : featureType_(FeatureType::NUMERICAL_OR_ORDINAL) {

        }

        FeatureType getFeatureType(uint32 featureIndex) const override {
            return featureType_;
        }

        void setAllNumerical() override {
            featureType_ = FeatureType::NUMERICAL_OR_ORDINAL;
        }

        void setAllBinary() override {
            featureType_ = FeatureType::BINARY;
        }

        void setAllNominal() override {
            featureType_ = FeatureType::NOMINAL;
        }

};

std::unique_ptr<IEqualFeatureInfo> createEqualFeatureInfo() {
    return std::make_unique<EqualFeatureInfo>();
}
