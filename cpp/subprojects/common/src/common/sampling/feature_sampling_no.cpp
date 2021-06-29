#include "common/sampling/feature_sampling_no.hpp"
#include "common/indices/index_vector_full.hpp"


/**
 * An implementation of the class `IFeatureSampling` that does not perform any sampling, but includes all features.
 */
class NoFeatureSubSampling final : public IFeatureSampling {

    private:

        FullIndexVector indexVector_;

    public:

        /**
         * @param numFeatures The total number of available features
         */
        NoFeatureSubSampling(uint32 numFeatures)
            : indexVector_(FullIndexVector(numFeatures)) {

        }

        const IIndexVector& subSample(RNG& rng) override {
            return indexVector_;
        }

};

std::unique_ptr<IFeatureSampling> NoFeatureSubSamplingFactory::create(uint32 numFeatures) const {
    return std::make_unique<NoFeatureSubSampling>(numFeatures);
}
