#include "common/sampling/feature_sampling_no.hpp"
#include "common/indices/index_vector_complete.hpp"


/**
 * An implementation of the class `IFeatureSampling` that does not perform any sampling, but includes all features.
 */
class NoFeatureSampling final : public IFeatureSampling {

    private:

        CompleteIndexVector indexVector_;

    public:

        /**
         * @param numFeatures The total number of available features
         */
        NoFeatureSampling(uint32 numFeatures)
            : indexVector_(CompleteIndexVector(numFeatures)) {

        }

        const IIndexVector& sample(RNG& rng) override {
            return indexVector_;
        }

};

/**
 * Allows to create instances of the type `IFeatureSampling` that do not perform any sampling, but include all features.
 */
class NoFeatureSamplingFactory final : public IFeatureSamplingFactory {

    public:

        std::unique_ptr<IFeatureSampling> create(uint32 numFeatures) const override {
            return std::make_unique<NoFeatureSampling>(numFeatures);
        }

};

std::unique_ptr<IFeatureSamplingFactory> NoFeatureSamplingConfig::create() const {
    return std::make_unique<NoFeatureSamplingFactory>();
}
