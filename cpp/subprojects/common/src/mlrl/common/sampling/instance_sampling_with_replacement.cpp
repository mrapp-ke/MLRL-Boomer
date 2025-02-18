#include "mlrl/common/sampling/instance_sampling_with_replacement.hpp"

#include "mlrl/common/sampling/partition_bi.hpp"
#include "mlrl/common/sampling/partition_single.hpp"
#include "mlrl/common/sampling/weight_vector_dense.hpp"
#include "mlrl/common/util/math.hpp"
#include "mlrl/common/util/validation.hpp"

template<typename ExampleWeights, typename WeightType>
static inline void sampleInternally(const SinglePartition& partition, const ExampleWeights& exampleWeights,
                                    float32 sampleSize, uint32 minSamples, uint32 maxSamples,
                                    DenseWeightVector<WeightType>& weightVector, RNG& rng) {
    uint32 numExamples = partition.getNumElements();
    uint32 numSamples = util::calculateBoundedFraction(numExamples, sampleSize, minSamples, maxSamples);
    typename DenseWeightVector<WeightType>::iterator weightIterator = weightVector.begin();
    util::setViewToZeros(weightIterator, numExamples);
    uint32 numNonZeroWeights = 0;

    for (uint32 i = 0; i < numSamples; i++) {
        // Randomly select the index of an example...
        uint32 randomIndex = rng.randomInt(0, numExamples);

        // Update weight at the selected index...
        typename ExampleWeights::weight_type exampleWeight = exampleWeights[randomIndex];
        WeightType previousWeight = weightIterator[randomIndex];
        weightIterator[randomIndex] = previousWeight + exampleWeight;

        if (previousWeight == 0) {
            numNonZeroWeights++;
        }
    }

    weightVector.setNumNonZeroWeights(numNonZeroWeights);
}

template<typename ExampleWeights, typename WeightType>
static inline void sampleInternally(BiPartition& partition, const ExampleWeights& exampleWeights, float32 sampleSize,
                                    uint32 minSamples, uint32 maxSamples, DenseWeightVector<WeightType>& weightVector,
                                    RNG& rng) {
    uint32 numExamples = partition.getNumElements();
    uint32 numTrainingExamples = partition.getNumFirst();
    uint32 numSamples = util::calculateBoundedFraction(numTrainingExamples, sampleSize, minSamples, maxSamples);
    BiPartition::const_iterator indexIterator = partition.first_cbegin();
    typename DenseWeightVector<WeightType>::iterator weightIterator = weightVector.begin();
    util::setViewToZeros(weightIterator, numExamples);
    uint32 numNonZeroWeights = 0;

    for (uint32 i = 0; i < numSamples; i++) {
        // Randomly select the index of an example...
        uint32 randomIndex = rng.randomInt(0, numTrainingExamples);
        uint32 sampledIndex = indexIterator[randomIndex];

        // Update weight at the selected index...
        typename ExampleWeights::weight_type exampleWeight = exampleWeights[sampledIndex];
        WeightType previousWeight = weightIterator[sampledIndex];
        weightIterator[sampledIndex] = previousWeight + exampleWeight;

        if (previousWeight == 0) {
            numNonZeroWeights++;
        }
    }

    weightVector.setNumNonZeroWeights(numNonZeroWeights);
}

/**
 * Allows to select a subset of the available training examples with replacement.
 *
 * @tparam Partition      The type of the object that provides access to the indices of the examples that are included
 *                        in the training set
 * @tparam ExampleWeights The type of the object that provides access to the weights of individual training examples
 * @tparam WeightType     The type of the weights of individual training examples in the sample
 */
template<typename Partition, typename ExampleWeights, typename WeightType>
class InstanceSamplingWithReplacement final : public IInstanceSampling {
    private:

        std::unique_ptr<RNG> rngPtr_;

        Partition& partition_;

        const ExampleWeights& exampleWeights_;

        const float32 sampleSize_;

        const uint32 minSamples_;

        const uint32 maxSamples_;

        DenseWeightVector<WeightType> weightVector_;

    public:

        /**
         * @param rngPtr          An unique pointer to an object of type `RNG` that should be used for generating random
         *                        numbers
         * @param partition       A reference to an object of template type `Partition` that provides access to the
         *                        indices of the examples that are included in the training set
         * @param exampleWeights  A reference to an object of template type `ExampleWeights` that provides access to the
         *                        weights of individual training examples
         * @param sampleSize      The fraction of examples to be included in the sample (e.g. a value of 0.6 corresponds
         *                        to 60 % of the available examples). Must be in (0, 1]
         * @param minSamples      The minimum number of examples to be included in the sample. Must be at least 1
         * @param maxSamples      The maximum number of examples to be included in the sample. Must be at least
         *                        `minSamples` or 0, if the number of examples should not be restricted
         */
        InstanceSamplingWithReplacement(std::unique_ptr<RNG> rngPtr, Partition& partition,
                                        const ExampleWeights& exampleWeights, float32 sampleSize, uint32 minSamples,
                                        uint32 maxSamples)
            : rngPtr_(std::move(rngPtr)), partition_(partition), exampleWeights_(exampleWeights),
              sampleSize_(sampleSize), minSamples_(minSamples), maxSamples_(maxSamples),
              weightVector_(partition.getNumElements()) {}

        const IWeightVector& sample() override {
            sampleInternally<ExampleWeights, WeightType>(partition_, exampleWeights_, sampleSize_, minSamples_,
                                                         maxSamples_, weightVector_, *rngPtr_);
            return weightVector_;
        }
};

/**
 * Allows to create instances of the type `IInstanceSampling` that allow to select a subset of the available training
 * examples with replacement.
 */
class InstanceSamplingWithReplacementFactory final : public IClassificationInstanceSamplingFactory,
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
         *                      to 60 % of the available examples). Must be in (0, 1]
         * @param minSamples    The minimum number of examples to be included in the sample. Must be at least 1
         * @param maxSamples    The maximum number of examples to be included in the sample. Must be at least
         *                      `minSamples` or 0, if the number of examples should not be restricted
         */
        InstanceSamplingWithReplacementFactory(std::unique_ptr<RNGFactory> rngFactoryPtr, float32 sampleSize,
                                               uint32 minSamples, uint32 maxSamples)
            : rngFactoryPtr_(std::move(rngFactoryPtr)), sampleSize_(sampleSize), minSamples_(minSamples),
              maxSamples_(maxSamples) {}

        std::unique_ptr<IInstanceSampling> create(const CContiguousView<const uint8>& labelMatrix,
                                                  const SinglePartition& partition, IStatistics& statistics,
                                                  const EqualWeightVector& exampleWeights) const override {
            return std::make_unique<InstanceSamplingWithReplacement<const SinglePartition, EqualWeightVector, uint32>>(
              rngFactoryPtr_->create(), partition, exampleWeights, sampleSize_, minSamples_, maxSamples_);
        }

        std::unique_ptr<IInstanceSampling> create(const CContiguousView<const uint8>& labelMatrix,
                                                  const SinglePartition& partition, IStatistics& statistics,
                                                  const DenseWeightVector<float32>& exampleWeights) const override {
            return std::make_unique<
              InstanceSamplingWithReplacement<const SinglePartition, DenseWeightVector<float32>, float32>>(
              rngFactoryPtr_->create(), partition, exampleWeights, sampleSize_, minSamples_, maxSamples_);
        }

        std::unique_ptr<IInstanceSampling> create(const CContiguousView<const uint8>& labelMatrix,
                                                  BiPartition& partition, IStatistics& statistics,
                                                  const EqualWeightVector& exampleWeights) const override {
            return std::make_unique<InstanceSamplingWithReplacement<BiPartition, EqualWeightVector, uint32>>(
              rngFactoryPtr_->create(), partition, exampleWeights, sampleSize_, minSamples_, maxSamples_);
        }

        std::unique_ptr<IInstanceSampling> create(const CContiguousView<const uint8>& labelMatrix,
                                                  BiPartition& partition, IStatistics& statistics,
                                                  const DenseWeightVector<float32>& exampleWeights) const override {
            return std::make_unique<InstanceSamplingWithReplacement<BiPartition, DenseWeightVector<float32>, float32>>(
              rngFactoryPtr_->create(), partition, exampleWeights, sampleSize_, minSamples_, maxSamples_);
        }

        std::unique_ptr<IInstanceSampling> create(const BinaryCsrView& labelMatrix, const SinglePartition& partition,
                                                  IStatistics& statistics,
                                                  const EqualWeightVector& exampleWeights) const override {
            return std::make_unique<InstanceSamplingWithReplacement<const SinglePartition, EqualWeightVector, uint32>>(
              rngFactoryPtr_->create(), partition, exampleWeights, sampleSize_, minSamples_, maxSamples_);
        }

        std::unique_ptr<IInstanceSampling> create(const BinaryCsrView& labelMatrix, const SinglePartition& partition,
                                                  IStatistics& statistics,
                                                  const DenseWeightVector<float32>& exampleWeights) const override {
            return std::make_unique<
              InstanceSamplingWithReplacement<const SinglePartition, DenseWeightVector<float32>, float32>>(
              rngFactoryPtr_->create(), partition, exampleWeights, sampleSize_, minSamples_, maxSamples_);
        }

        std::unique_ptr<IInstanceSampling> create(const BinaryCsrView& labelMatrix, BiPartition& partition,
                                                  IStatistics& statistics,
                                                  const EqualWeightVector& exampleWeights) const override {
            return std::make_unique<InstanceSamplingWithReplacement<BiPartition, EqualWeightVector, uint32>>(
              rngFactoryPtr_->create(), partition, exampleWeights, sampleSize_, minSamples_, maxSamples_);
        }

        std::unique_ptr<IInstanceSampling> create(const BinaryCsrView& labelMatrix, BiPartition& partition,
                                                  IStatistics& statistics,
                                                  const DenseWeightVector<float32>& exampleWeights) const override {
            return std::make_unique<InstanceSamplingWithReplacement<BiPartition, DenseWeightVector<float32>, float32>>(
              rngFactoryPtr_->create(), partition, exampleWeights, sampleSize_, minSamples_, maxSamples_);
        }

        std::unique_ptr<IInstanceSampling> create(const CContiguousView<const float32>& regressionMatrix,
                                                  const SinglePartition& partition, IStatistics& statistics,
                                                  const EqualWeightVector& exampleWeights) const override {
            return std::make_unique<InstanceSamplingWithReplacement<const SinglePartition, EqualWeightVector, uint32>>(
              rngFactoryPtr_->create(), partition, exampleWeights, sampleSize_, minSamples_, maxSamples_);
        }

        std::unique_ptr<IInstanceSampling> create(const CContiguousView<const float32>& regressionMatrix,
                                                  const SinglePartition& partition, IStatistics& statistics,
                                                  const DenseWeightVector<float32>& exampleWeights) const override {
            return std::make_unique<
              InstanceSamplingWithReplacement<const SinglePartition, DenseWeightVector<float32>, float32>>(
              rngFactoryPtr_->create(), partition, exampleWeights, sampleSize_, minSamples_, maxSamples_);
        }

        std::unique_ptr<IInstanceSampling> create(const CContiguousView<const float32>& regressionMatrix,
                                                  BiPartition& partition, IStatistics& statistics,
                                                  const EqualWeightVector& exampleWeights) const override {
            return std::make_unique<InstanceSamplingWithReplacement<BiPartition, EqualWeightVector, uint32>>(
              rngFactoryPtr_->create(), partition, exampleWeights, sampleSize_, minSamples_, maxSamples_);
        }

        std::unique_ptr<IInstanceSampling> create(const CContiguousView<const float32>& regressionMatrix,
                                                  BiPartition& partition, IStatistics& statistics,
                                                  const DenseWeightVector<float32>& exampleWeights) const override {
            return std::make_unique<InstanceSamplingWithReplacement<BiPartition, DenseWeightVector<float32>, float32>>(
              rngFactoryPtr_->create(), partition, exampleWeights, sampleSize_, minSamples_, maxSamples_);
        }

        std::unique_ptr<IInstanceSampling> create(const CsrView<const float32>& regressionMatrix,
                                                  const SinglePartition& partition, IStatistics& statistics,
                                                  const EqualWeightVector& exampleWeights) const override {
            return std::make_unique<InstanceSamplingWithReplacement<const SinglePartition, EqualWeightVector, uint32>>(
              rngFactoryPtr_->create(), partition, exampleWeights, sampleSize_, minSamples_, maxSamples_);
        }

        std::unique_ptr<IInstanceSampling> create(const CsrView<const float32>& regressionMatrix,
                                                  const SinglePartition& partition, IStatistics& statistics,
                                                  const DenseWeightVector<float32>& exampleWeights) const override {
            return std::make_unique<
              InstanceSamplingWithReplacement<const SinglePartition, DenseWeightVector<float32>, float32>>(
              rngFactoryPtr_->create(), partition, exampleWeights, sampleSize_, minSamples_, maxSamples_);
        }

        std::unique_ptr<IInstanceSampling> create(const CsrView<const float32>& regressionMatrix,
                                                  BiPartition& partition, IStatistics& statistics,
                                                  const EqualWeightVector& exampleWeights) const override {
            return std::make_unique<InstanceSamplingWithReplacement<BiPartition, EqualWeightVector, uint32>>(
              rngFactoryPtr_->create(), partition, exampleWeights, sampleSize_, minSamples_, maxSamples_);
        }

        std::unique_ptr<IInstanceSampling> create(const CsrView<const float32>& regressionMatrix,
                                                  BiPartition& partition, IStatistics& statistics,
                                                  const DenseWeightVector<float32>& exampleWeights) const override {
            return std::make_unique<InstanceSamplingWithReplacement<BiPartition, DenseWeightVector<float32>, float32>>(
              rngFactoryPtr_->create(), partition, exampleWeights, sampleSize_, minSamples_, maxSamples_);
        }
};

InstanceSamplingWithReplacementConfig::InstanceSamplingWithReplacementConfig(ReadableProperty<RNGConfig> rngConfig)
    : rngConfig_(rngConfig), sampleSize_(0.66f), minSamples_(1), maxSamples_(0) {}

float32 InstanceSamplingWithReplacementConfig::getSampleSize() const {
    return sampleSize_;
}

IInstanceSamplingWithReplacementConfig& InstanceSamplingWithReplacementConfig::setSampleSize(float32 sampleSize) {
    util::assertGreater<float32>("sampleSize", sampleSize, 0);
    util::assertLessOrEqual<float32>("sampleSize", sampleSize, 1);
    sampleSize_ = sampleSize;
    return *this;
}

uint32 InstanceSamplingWithReplacementConfig::getMinSamples() const {
    return minSamples_;
}

IInstanceSamplingWithReplacementConfig& InstanceSamplingWithReplacementConfig::setMinSamples(uint32 minSamples) {
    util::assertGreaterOrEqual<uint32>("minSamples", minSamples, 1);
    minSamples_ = minSamples;
    return *this;
}

uint32 InstanceSamplingWithReplacementConfig::getMaxSamples() const {
    return maxSamples_;
}

IInstanceSamplingWithReplacementConfig& InstanceSamplingWithReplacementConfig::setMaxSamples(uint32 maxSamples) {
    if (maxSamples != 0) util::assertGreaterOrEqual<uint32>("maxSamples", maxSamples, minSamples_);
    maxSamples_ = maxSamples;
    return *this;
}

std::unique_ptr<IClassificationInstanceSamplingFactory>
  InstanceSamplingWithReplacementConfig::createClassificationInstanceSamplingFactory() const {
    return std::make_unique<InstanceSamplingWithReplacementFactory>(rngConfig_.get().createRNGFactory(), sampleSize_,
                                                                    minSamples_, maxSamples_);
}

std::unique_ptr<IRegressionInstanceSamplingFactory>
  InstanceSamplingWithReplacementConfig::createRegressionInstanceSamplingFactory() const {
    return std::make_unique<InstanceSamplingWithReplacementFactory>(rngConfig_.get().createRNGFactory(), sampleSize_,
                                                                    minSamples_, maxSamples_);
}
