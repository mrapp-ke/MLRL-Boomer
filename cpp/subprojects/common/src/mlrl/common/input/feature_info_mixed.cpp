#include "mlrl/common/input/feature_info_mixed.hpp"

#include "mlrl/common/data/vector_bit.hpp"
#include "mlrl/common/input/feature_type_nominal.hpp"
#include "mlrl/common/input/feature_type_numerical.hpp"
#include "mlrl/common/input/feature_type_ordinal.hpp"

/**
 * An implementation of the type `IMixedFeatureInfo` that uses `BitVector`s to store whether individual features are
 * ordinal, nominal or numerical.
 */
class BitFeatureInfo final : public IMixedFeatureInfo {
    private:

        BitVector ordinalBitVector_;

        BitVector nominalBitVector_;

    public:

        /**
         * @param numFeatures The total number of available features
         */
        BitFeatureInfo(uint32 numFeatures)
            : ordinalBitVector_(numFeatures, true), nominalBitVector_(numFeatures, true) {}

        std::unique_ptr<IFeatureType> createFeatureType(
          uint32 featureIndex, const IFeatureBinningFactory& featureBinningFactory) const override {
            if (ordinalBitVector_[featureIndex]) {
                return std::make_unique<OrdinalFeatureType>();
            } else if (nominalBitVector_[featureIndex]) {
                return std::make_unique<NominalFeatureType>();
            } else {
                std::unique_ptr<IFeatureBinning> featureBinningPtr = featureBinningFactory.create();

                if (featureBinningPtr) {
                    return featureBinningPtr;
                }

                return std::make_unique<NumericalFeatureType>();
            }
        }

        void setNumerical(uint32 featureIndex) override {
            ordinalBitVector_.set(featureIndex, false);
            nominalBitVector_.set(featureIndex, false);
        }

        void setOrdinal(uint32 featureIndex) override {
            ordinalBitVector_.set(featureIndex, true);
            nominalBitVector_.set(featureIndex, false);
        }

        void setNominal(uint32 featureIndex) override {
            ordinalBitVector_.set(featureIndex, false);
            nominalBitVector_.set(featureIndex, true);
        }
};

std::unique_ptr<IMixedFeatureInfo> createMixedFeatureInfo(uint32 numFeatures) {
    return std::make_unique<BitFeatureInfo>(numFeatures);
}
