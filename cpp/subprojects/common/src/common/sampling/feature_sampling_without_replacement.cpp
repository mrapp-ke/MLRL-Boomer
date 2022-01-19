#include "common/sampling/feature_sampling_without_replacement.hpp"
#include "common/indices/index_vector_partial.hpp"
#include "common/iterator/index_iterator.hpp"
#include "common/util/validation.hpp"
#include "index_sampling.hpp"
#include <cmath>


/**
 * Allows to select a subset of the available features without replacement.
 */
class FeatureSamplingWithoutReplacement final : public IFeatureSampling {

    private:

        uint32 numFeatures_;

        PartialIndexVector indexVector_;

    public:

        /**
         * @param numFeatures   The total number of available features
         * @param sampleSize    The fraction of features to be included in the sample (e.g. a value of 0.6 corresponds
         *                      to 60 % of the available features). Must be in (0, 1) or 0, if the default sample size
         *                      `floor(log2(num_features - 1) + 1)` should be used
         */
        FeatureSamplingWithoutReplacement(uint32 numFeatures, float32 sampleSize)
            : numFeatures_(numFeatures),
              indexVector_(PartialIndexVector((uint32) (sampleSize > 0 ? sampleSize * numFeatures
                                                                       : log2(numFeatures - 1) + 1))) {

        }

        const IIndexVector& sample(RNG& rng) override {
            sampleIndicesWithoutReplacement<IndexIterator>(indexVector_, IndexIterator(numFeatures_), numFeatures_,
                                                           rng);
            return indexVector_;
        }

};

/**
 * Allows to create instances of the type `IFeatureSampling` that select a random subset of the available features
 * without replacement.
 */
class FeatureSamplingWithoutReplacementFactory final : public IFeatureSamplingFactory {

    private:

        float32 sampleSize_;

    public:

        /**
         * @param sampleSize The fraction of features to be included in the sample (e.g. a value of 0.6 corresponds to
         *                   60 % of the available features). Must be in (0, 1) or 0, if the default sample size
         *                   `floor(log2(num_features - 1) + 1)` should be used
         */
        FeatureSamplingWithoutReplacementFactory(float32 sampleSize)
            : sampleSize_(sampleSize) {

        }

        std::unique_ptr<IFeatureSampling> create(uint32 numFeatures) const override {
            return std::make_unique<FeatureSamplingWithoutReplacement>(numFeatures, sampleSize_);
        }

};

FeatureSamplingWithoutReplacementConfig::FeatureSamplingWithoutReplacementConfig()
    : sampleSize_(0) {

}

float32 FeatureSamplingWithoutReplacementConfig::getSampleSize() const {
    return sampleSize_;
}

IFeatureSamplingWithoutReplacementConfig& FeatureSamplingWithoutReplacementConfig::setSampleSize(float32 sampleSize) {
    assertGreaterOrEqual<float32>("sampleSize", sampleSize, 0);
    assertLess<float32>("sampleSize", sampleSize, 1);
    sampleSize_ = sampleSize;
    return *this;
}

std::unique_ptr<IFeatureSamplingFactory> FeatureSamplingWithoutReplacementConfig::create() const {
    return std::make_unique<FeatureSamplingWithoutReplacementFactory>(sampleSize_);
}
