#include "mlrl/common/sampling/feature_sampling_without_replacement.hpp"

#include "index_sampling.hpp"
#include "mlrl/common/indices/index_vector_partial.hpp"
#include "mlrl/common/iterator/iterator_index.hpp"
#include "mlrl/common/sampling/feature_sampling_predefined.hpp"
#include "mlrl/common/util/math.hpp"
#include "mlrl/common/util/validation.hpp"

/**
 * Allows to select a subset of the available features without replacement.
 */
class FeatureSamplingWithoutReplacement final : public IFeatureSampling {
    private:

        const std::shared_ptr<RNG> rngPtr_;

        const uint32 numFeatures_;

        const uint32 numSamples_;

        const uint32 numRetained_;

        PartialIndexVector indexVector_;

    public:

        /**
         * @param rngPtr        A shared pointer to an object of type `RNG` that should be used for generating random
         *                      numbers
         * @param numFeatures   The total number of available features
         * @param numSamples    The number of features to be included in the sample
         * @param numRetained   The number of trailing features to be always included in the sample
         */
        FeatureSamplingWithoutReplacement(std::shared_ptr<RNG> rngPtr, uint32 numFeatures, uint32 numSamples,
                                          uint32 numRetained)
            : rngPtr_(rngPtr), numFeatures_(numFeatures), numSamples_(numSamples), numRetained_(numRetained),
              indexVector_(numSamples + numRetained) {
            if (numRetained > 0) {
                PartialIndexVector::iterator iterator = indexVector_.begin();
                uint32 offset = numFeatures - numRetained;

                for (uint32 i = 0; i < numRetained; i++) {
                    iterator[i] = offset + i;
                }
            }
        }

        const IIndexVector& sample() override {
            uint32 numTotal = numFeatures_ - numRetained_;
            sampleIndicesWithoutReplacement<IndexIterator>(&indexVector_.begin()[numRetained_], numSamples_,
                                                           IndexIterator(numTotal), numTotal, *rngPtr_);
            return indexVector_;
        }

        std::unique_ptr<IFeatureSampling> createBeamSearchFeatureSampling(bool resample) override {
            if (resample) {
                return std::make_unique<FeatureSamplingWithoutReplacement>(rngPtr_, numFeatures_, numSamples_,
                                                                           numRetained_);
            } else {
                return std::make_unique<PredefinedFeatureSampling>(this->sample());
            }
        }
};

/**
 * Allows to create instances of the type `IFeatureSampling` that select a random subset of the available features
 * without replacement.
 */
class FeatureSamplingWithoutReplacementFactory final : public IFeatureSamplingFactory {
    private:

        const std::unique_ptr<RNGFactory> rngFactoryPtr_;

        const uint32 numFeatures_;

        const uint32 numSamples_;

        const uint32 numRetained_;

    public:

        /**
         * @param rngFactoryPtr An unique pointer to an object of type `RNGFactory` that allows to create random number
         *                      generators
         * @param numFeatures   The total number of available features
         * @param numSamples    The number of features to be included in the sample
         * @param numRetained   The number of trailing features to be always included in the sample
         */
        FeatureSamplingWithoutReplacementFactory(std::unique_ptr<RNGFactory> rngFactoryPtr, uint32 numFeatures,
                                                 uint32 numSamples, uint32 numRetained)
            : rngFactoryPtr_(std::move(rngFactoryPtr)), numFeatures_(numFeatures), numSamples_(numSamples),
              numRetained_(numRetained) {}

        std::unique_ptr<IFeatureSampling> create() const override {
            std::shared_ptr<RNG> rngPtr = rngFactoryPtr_->create();
            return std::make_unique<FeatureSamplingWithoutReplacement>(rngPtr, numFeatures_, numSamples_, numRetained_);
        }
};

FeatureSamplingWithoutReplacementConfig::FeatureSamplingWithoutReplacementConfig(ReadableProperty<RNGConfig> rngConfig)
    : rngConfig_(rngConfig), sampleSize_(0), minSamples_(1), maxSamples_(0), numRetained_(0) {}

float32 FeatureSamplingWithoutReplacementConfig::getSampleSize() const {
    return sampleSize_;
}

IFeatureSamplingWithoutReplacementConfig& FeatureSamplingWithoutReplacementConfig::setSampleSize(float32 sampleSize) {
    util::assertGreaterOrEqual<float32>("sampleSize", sampleSize, 0);
    util::assertLess<float32>("sampleSize", sampleSize, 1);
    sampleSize_ = sampleSize;
    return *this;
}

uint32 FeatureSamplingWithoutReplacementConfig::getMinSamples() const {
    return minSamples_;
}

IFeatureSamplingWithoutReplacementConfig& FeatureSamplingWithoutReplacementConfig::setMinSamples(uint32 minSamples) {
    util::assertGreaterOrEqual<uint32>("minSamples", minSamples, 1);
    minSamples_ = minSamples;
    return *this;
}

uint32 FeatureSamplingWithoutReplacementConfig::getMaxSamples() const {
    return maxSamples_;
}

IFeatureSamplingWithoutReplacementConfig& FeatureSamplingWithoutReplacementConfig::setMaxSamples(uint32 maxSamples) {
    if (maxSamples != 0) util::assertGreaterOrEqual<uint32>("maxSamples", maxSamples, minSamples_);
    maxSamples_ = maxSamples;
    return *this;
}

uint32 FeatureSamplingWithoutReplacementConfig::getNumRetained() const {
    return numRetained_;
}

IFeatureSamplingWithoutReplacementConfig& FeatureSamplingWithoutReplacementConfig::setNumRetained(uint32 numRetained) {
    util::assertGreaterOrEqual<uint32>("numRetained", numRetained, 0);
    numRetained_ = numRetained;
    return *this;
}

std::unique_ptr<IFeatureSamplingFactory> FeatureSamplingWithoutReplacementConfig::createFeatureSamplingFactory(
  const IFeatureMatrix& featureMatrix) const {
    uint32 numFeatures = featureMatrix.getNumFeatures();
    uint32 numRetained = std::min(numRetained_, numFeatures);
    uint32 numRemainingFeatures = numFeatures - numRetained;
    uint32 numSamples;

    if (sampleSize_ > 0) {
        numSamples = util::calculateBoundedFraction(numRemainingFeatures, sampleSize_, minSamples_, maxSamples_);
    } else {
        numSamples = static_cast<uint32>(log2(numRemainingFeatures - 1) + 1);
    }

    return std::make_unique<FeatureSamplingWithoutReplacementFactory>(rngConfig_.get().createRNGFactory(), numFeatures,
                                                                      numSamples, numRetained);
}

bool FeatureSamplingWithoutReplacementConfig::isSamplingUsed() const {
    return true;
}
