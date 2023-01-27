#include "common/input/feature_info_mixed.hpp"
#include "common/input/feature_type_binary.hpp"
#include "common/input/feature_type_nominal.hpp"
#include "common/input/feature_type_numerical.hpp"

/**
 * An implementation of the type `IMixedFeatureInfo` that uses `BitVector`s to store whether individual features are
 * binary, nominal or numerical/ordinal.
 */
class BitFeatureInfo final : public IMixedFeatureInfo {
    private:

        BitVector binaryBitVector_;

        BitVector nominalBitVector_;

    public:

        /**
         * @param numFeatures The total number of available features
         */
        BitFeatureInfo(uint32 numFeatures)
            : binaryBitVector_(BitVector(numFeatures, true)), nominalBitVector_(BitVector(numFeatures, true)) {

        }

        std::unique_ptr<IFeatureType> createFeatureType(uint32 featureIndex) const override {
            if (binaryBitVector_[featureIndex]) {
                return std::make_unique<BinaryFeatureType>();
            } else if (nominalBitVector_[featureIndex]) {
                return std::make_unique<NominalFeatureType>();
            } else {
                return std::make_unique<NumericalFeatureType>();
            }
        }

        void setNumerical(uint32 featureIndex) override {
            binaryBitVector_.set(featureIndex, false);
            nominalBitVector_.set(featureIndex, false);
        }

        void setBinary(uint32 featureIndex) override {
            binaryBitVector_.set(featureIndex, true);
            nominalBitVector_.set(featureIndex, false);
        }

        void setNominal(uint32 featureIndex) override {
            binaryBitVector_.set(featureIndex, false);
            nominalBitVector_.set(featureIndex, true);
        }
};

std::unique_ptr<IMixedFeatureInfo> createMixedFeatureInfo(uint32 numFeatures) {
    return std::make_unique<BitFeatureInfo>(numFeatures);
}
