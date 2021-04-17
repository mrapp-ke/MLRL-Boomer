#include "common/sampling/feature_sampling_no.hpp"
#include "common/indices/index_vector_full.hpp"


/**
 * An implementation of the class `IFeatureSubSampling` that does not perform any sampling, but includes all features.
 */
class NoFeatureSubSampling final : public IFeatureSubSampling {

    private:

        uint32 numFeatures_;

    public:

        /**
         * @param numFeatures The total number of available features
         */
        NoFeatureSubSampling(uint32 numFeatures)
            : numFeatures_(numFeatures) {

        }

        std::unique_ptr<IIndexVector> subSample(RNG& rng) const override {
            return std::make_unique<FullIndexVector>(numFeatures_);
        }

};

std::unique_ptr<IFeatureSubSampling> NoFeatureSubSamplingFactory::create(uint32 numFeatures) const {
    return std::make_unique<NoFeatureSubSampling>(numFeatures);
}
