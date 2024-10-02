#include "mlrl/common/sampling/instance_sampling_without_replacement.hpp"

#include "mlrl/common/iterator/iterator_index.hpp"
#include "mlrl/common/sampling/partition_bi.hpp"
#include "mlrl/common/sampling/partition_single.hpp"
#include "mlrl/common/sampling/weight_sampling.hpp"
#include "mlrl/common/util/math.hpp"
#include "mlrl/common/util/validation.hpp"

static inline void sampleInternally(const SinglePartition& partition, float32 sampleSize, uint32 minSamples,
                                    uint32 maxSamples, BitWeightVector& weightVector, RNG& rng) {
    uint32 numExamples = partition.getNumElements();
    uint32 numSamples = util::calculateBoundedFraction(numExamples, sampleSize, minSamples, maxSamples);
    sampleWeightsWithoutReplacement<SinglePartition::const_iterator>(weightVector, partition.cbegin(), numExamples,
                                                                     numSamples, rng);
}

static inline void sampleInternally(BiPartition& partition, float32 sampleSize, uint32 minSamples, uint32 maxSamples,
                                    BitWeightVector& weightVector, RNG& rng) {
    uint32 numTrainingExamples = partition.getNumFirst();
    uint32 numSamples = util::calculateBoundedFraction(numTrainingExamples, sampleSize, minSamples, maxSamples);
    sampleWeightsWithoutReplacement<BiPartition::const_iterator>(weightVector, partition.first_cbegin(),
                                                                 numTrainingExamples, numSamples, rng);
}

/**
 * Allows to select a subset of the available training examples without replacement.
 *
 * @tparam Partition The type of the object that provides access to the indices of the examples that are included in the
 *                   training set
 */
template<typename Partition>
class InstanceSamplingWithoutReplacement final : public IInstanceSampling {
    private:

        std::unique_ptr<RNG> rngPtr_;

        Partition& partition_;

        const float32 sampleSize_;

        const uint32 minSamples_;

        const uint32 maxSamples_;

        BitWeightVector weightVector_;

    public:

        /**
         * @param rngPtr        An unique pointer to an object of type `RNG` that should be used for generating random
         *                      numbers
         * @param partition     A reference to an object of template type `Partition` that provides access to the
         *                      indices of the examples that are included in the training set
         * @param sampleSize    The fraction of examples to be included in the sample (e.g. a value of 0.6 corresponds
         *                      to 60 % of the available examples). Must be in (0, 1)
         * @param minSamples    The minimum number of examples to be included in the sample. Must be at least 1
         * @param maxSamples    The maximum number of examples to be included in the sample. Must be at least
         *                      `minSamples` or 0, if the number of examples should not be restricted
         */
        InstanceSamplingWithoutReplacement(std::unique_ptr<RNG> rngPtr, Partition& partition, float32 sampleSize,
                                           uint32 minSamples, uint32 maxSamples)
            : rngPtr_(std::move(rngPtr)), partition_(partition), sampleSize_(sampleSize), minSamples_(minSamples),
              maxSamples_(maxSamples), weightVector_(partition.getNumElements()) {}

        const IWeightVector& sample() override {
            sampleInternally(partition_, sampleSize_, minSamples_, maxSamples_, weightVector_, *rngPtr_);
            return weightVector_;
        }
};

/**
 * Allows to create instances of the type `IInstanceSampling` that allow to select a subset of the available training
 * examples without replacement.
 */
class InstanceSamplingWithoutReplacementFactory final : public IClassificationInstanceSamplingFactory,
                                                        public IRegressionInstanceSamplingFactory {
    private:

        const std::unique_ptr<RNGFactory> rngFactoryPtr_;

        const float32 sampleSize_;

        const uint32 minSamples_;

        const uint32 maxSamples_;

    public:

        /**
         * @param rngFactoryPtr An unique pointer to an object of type `RNGFactory` that allows to create random number
         *                      generators
         * @param sampleSize    The fraction of examples to be included in the sample (e.g. a value of 0.6 corresponds
         *                      to 60 % of the available examples). Must be in (0, 1)
         * @param minSamples    The minimum number of examples to be included in the sample. Must be at least 1
         * @param maxSamples    The maximum number of examples to be included in the sample. Must be at least
         *                      `minSamples` or 0, if the number of examples should not be restricted
         */
        InstanceSamplingWithoutReplacementFactory(std::unique_ptr<RNGFactory> rngFactoryPtr, float32 sampleSize,
                                                  uint32 minSamples, uint32 maxSamples)
            : rngFactoryPtr_(std::move(rngFactoryPtr)), sampleSize_(sampleSize), minSamples_(minSamples),
              maxSamples_(maxSamples) {}

        std::unique_ptr<IInstanceSampling> create(const CContiguousView<const uint8>& labelMatrix,
                                                  const SinglePartition& partition,
                                                  IStatistics& statistics) const override {
            return std::make_unique<InstanceSamplingWithoutReplacement<const SinglePartition>>(
              rngFactoryPtr_->create(), partition, sampleSize_, minSamples_, maxSamples_);
        }

        std::unique_ptr<IInstanceSampling> create(const CContiguousView<const uint8>& labelMatrix,
                                                  BiPartition& partition, IStatistics& statistics) const override {
            return std::make_unique<InstanceSamplingWithoutReplacement<BiPartition>>(
              rngFactoryPtr_->create(), partition, sampleSize_, minSamples_, maxSamples_);
        }

        std::unique_ptr<IInstanceSampling> create(const BinaryCsrView& labelMatrix, const SinglePartition& partition,
                                                  IStatistics& statistics) const override {
            return std::make_unique<InstanceSamplingWithoutReplacement<const SinglePartition>>(
              rngFactoryPtr_->create(), partition, sampleSize_, minSamples_, maxSamples_);
        }

        std::unique_ptr<IInstanceSampling> create(const BinaryCsrView& labelMatrix, BiPartition& partition,
                                                  IStatistics& statistics) const override {
            return std::make_unique<InstanceSamplingWithoutReplacement<BiPartition>>(
              rngFactoryPtr_->create(), partition, sampleSize_, minSamples_, maxSamples_);
        }

        std::unique_ptr<IInstanceSampling> create(const CContiguousView<const float32>& regressionMatrix,
                                                  const SinglePartition& partition,
                                                  IStatistics& statistics) const override {
            return std::make_unique<InstanceSamplingWithoutReplacement<const SinglePartition>>(
              rngFactoryPtr_->create(), partition, sampleSize_, minSamples_, maxSamples_);
        }

        std::unique_ptr<IInstanceSampling> create(const CContiguousView<const float32>& regressionMatrix,
                                                  BiPartition& partition, IStatistics& statistics) const override {
            return std::make_unique<InstanceSamplingWithoutReplacement<BiPartition>>(
              rngFactoryPtr_->create(), partition, sampleSize_, minSamples_, maxSamples_);
        }

        std::unique_ptr<IInstanceSampling> create(const CsrView<const float32>& regressionMatrix,
                                                  const SinglePartition& partition,
                                                  IStatistics& statistics) const override {
            return std::make_unique<InstanceSamplingWithoutReplacement<const SinglePartition>>(
              rngFactoryPtr_->create(), partition, sampleSize_, minSamples_, maxSamples_);
        }

        std::unique_ptr<IInstanceSampling> create(const CsrView<const float32>& regressionMatrix,
                                                  BiPartition& partition, IStatistics& statistics) const override {
            return std::make_unique<InstanceSamplingWithoutReplacement<BiPartition>>(
              rngFactoryPtr_->create(), partition, sampleSize_, minSamples_, maxSamples_);
        }
};

InstanceSamplingWithoutReplacementConfig::InstanceSamplingWithoutReplacementConfig(
  ReadableProperty<RNGConfig> rngConfig)
    : rngConfig_(rngConfig), sampleSize_(0.66f), minSamples_(1), maxSamples_(0) {}

float32 InstanceSamplingWithoutReplacementConfig::getSampleSize() const {
    return sampleSize_;
}

IInstanceSamplingWithoutReplacementConfig& InstanceSamplingWithoutReplacementConfig::setSampleSize(float32 sampleSize) {
    util::assertGreater<float32>("sampleSize", sampleSize, 0);
    util::assertLess<float32>("sampleSize", sampleSize, 1);
    sampleSize_ = sampleSize;
    return *this;
}

uint32 InstanceSamplingWithoutReplacementConfig::getMinSamples() const {
    return minSamples_;
}

IInstanceSamplingWithoutReplacementConfig& InstanceSamplingWithoutReplacementConfig::setMinSamples(uint32 minSamples) {
    util::assertGreaterOrEqual<uint32>("minSamples", minSamples, 1);
    minSamples_ = minSamples;
    return *this;
}

uint32 InstanceSamplingWithoutReplacementConfig::getMaxSamples() const {
    return maxSamples_;
}

IInstanceSamplingWithoutReplacementConfig& InstanceSamplingWithoutReplacementConfig::setMaxSamples(uint32 maxSamples) {
    if (maxSamples != 0) util::assertGreaterOrEqual<uint32>("maxSamples", maxSamples, minSamples_);
    maxSamples_ = maxSamples;
    return *this;
}

std::unique_ptr<IClassificationInstanceSamplingFactory>
  InstanceSamplingWithoutReplacementConfig::createClassificationInstanceSamplingFactory() const {
    return std::make_unique<InstanceSamplingWithoutReplacementFactory>(rngConfig_.get().createRNGFactory(), sampleSize_,
                                                                       minSamples_, maxSamples_);
}

std::unique_ptr<IRegressionInstanceSamplingFactory>
  InstanceSamplingWithoutReplacementConfig::createRegressionInstanceSamplingFactory() const {
    return std::make_unique<InstanceSamplingWithoutReplacementFactory>(rngConfig_.get().createRNGFactory(), sampleSize_,
                                                                       minSamples_, maxSamples_);
}
