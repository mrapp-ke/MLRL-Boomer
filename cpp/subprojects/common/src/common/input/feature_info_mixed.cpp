#include "common/input/feature_info_mixed.hpp"
#include "common/data/vector_bit.hpp"


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

        FeatureType getFeatureType(uint32 featureIndex) const override {
            if (binaryBitVector_[featureIndex]) {
                return FeatureType::BINARY;
            } else if (nominalBitVector_[featureIndex]) {
                return FeatureType::NOMINAL;
            } else {
                return FeatureType::NUMERICAL_OR_ORDINAL;
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
