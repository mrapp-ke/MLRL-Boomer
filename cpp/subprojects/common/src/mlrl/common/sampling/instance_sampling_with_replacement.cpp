#include "mlrl/common/sampling/instance_sampling_with_replacement.hpp"

#include "mlrl/common/sampling/partition_bi.hpp"
#include "mlrl/common/sampling/partition_single.hpp"
#include "mlrl/common/sampling/weight_vector_dense.hpp"
#include "mlrl/common/util/validation.hpp"

static inline void sampleInternally(const SinglePartition& partition, float32 sampleSize,
                                    DenseWeightVector<uint32>& weightVector, RNG& rng) {
    uint32 numExamples = partition.getNumElements();
    uint32 numSamples = static_cast<uint32>(sampleSize * numExamples);
    typename DenseWeightVector<uint32>::iterator weightIterator = weightVector.begin();
    setViewToZeros(weightIterator, numExamples);
    uint32 numNonZeroWeights = 0;

    for (uint32 i = 0; i < numSamples; i++) {
        // Randomly select the index of an example...
        uint32 randomIndex = rng.random(0, numExamples);

        // Update weight at the selected index...
        uint32 previousWeight = weightIterator[randomIndex];
        weightIterator[randomIndex] = previousWeight + 1;

        if (previousWeight == 0) {
            numNonZeroWeights++;
        }
    }

    weightVector.setNumNonZeroWeights(numNonZeroWeights);
}

static inline void sampleInternally(BiPartition& partition, float32 sampleSize, DenseWeightVector<uint32>& weightVector,
                                    RNG& rng) {
    uint32 numExamples = partition.getNumElements();
    uint32 numTrainingExamples = partition.getNumFirst();
    uint32 numSamples = static_cast<uint32>(sampleSize * numTrainingExamples);
    BiPartition::const_iterator indexIterator = partition.first_cbegin();
    typename DenseWeightVector<uint32>::iterator weightIterator = weightVector.begin();
    setViewToZeros(weightIterator, numExamples);
    uint32 numNonZeroWeights = 0;

    for (uint32 i = 0; i < numSamples; i++) {
        // Randomly select the index of an example...
        uint32 randomIndex = rng.random(0, numTrainingExamples);
        uint32 sampledIndex = indexIterator[randomIndex];

        // Update weight at the selected index...
        uint32 previousWeight = weightIterator[sampledIndex];
        weightIterator[sampledIndex] = previousWeight + 1;

        if (previousWeight == 0) {
            numNonZeroWeights++;
        }
    }

    weightVector.setNumNonZeroWeights(numNonZeroWeights);
}

/**
 * Allows to select a subset of the available training examples with replacement.
 *
 * @tparam Partition The type of the object that provides access to the indices of the examples that are included in the
 *                   training set
 */
template<typename Partition>
class InstanceSamplingWithReplacement final : public IInstanceSampling {
    private:

        Partition& partition_;

        const float32 sampleSize_;

        DenseWeightVector<uint32> weightVector_;

    public:

        /**
         * @param partition  A reference to an object of template type `Partition` that provides access to the indices
         *                   of the examples that are included in the training set
         * @param sampleSize The fraction of examples to be included in the sample (e.g. a value of 0.6 corresponds to
         *                   60 % of the available examples). Must be in (0, 1]
         */
        InstanceSamplingWithReplacement(Partition& partition, float32 sampleSize)
            : partition_(partition), sampleSize_(sampleSize), weightVector_(partition.getNumElements()) {}

        const IWeightVector& sample(RNG& rng) override {
            sampleInternally(partition_, sampleSize_, weightVector_, rng);
            return weightVector_;
        }
};

template<typename Partition>
static inline std::unique_ptr<IInstanceSampling> createInstanceSamplingWithReplacement(Partition& partition,
                                                                                       float32 sampleSize) {
    return std::make_unique<InstanceSamplingWithReplacement<Partition>>(partition, sampleSize);
}

/**
 * Allows to create instances of the type `IInstanceSampling` that allow to select a subset of the available training
 * examples with replacement.
 */
class InstanceSamplingWithReplacementFactory final : public IClassificationInstanceSamplingFactory,
                                                     public IRegressionInstanceSamplingFactory {
    private:

        const float32 sampleSize_;

    public:

        /**
         * @param sampleSize The fraction of examples to be included in the sample (e.g. a value of 0.6 corresponds to
         *                   60 % of the available examples). Must be in (0, 1]
         */
        InstanceSamplingWithReplacementFactory(float32 sampleSize) : sampleSize_(sampleSize) {}

        std::unique_ptr<IInstanceSampling> create(const CContiguousView<const uint8>& labelMatrix,
                                                  const SinglePartition& partition,
                                                  IStatistics& statistics) const override {
            return createInstanceSamplingWithReplacement(partition, sampleSize_);
        }

        std::unique_ptr<IInstanceSampling> create(const CContiguousView<const uint8>& labelMatrix,
                                                  BiPartition& partition, IStatistics& statistics) const override {
            return createInstanceSamplingWithReplacement(partition, sampleSize_);
        }

        std::unique_ptr<IInstanceSampling> create(const BinaryCsrView& labelMatrix, const SinglePartition& partition,
                                                  IStatistics& statistics) const override {
            return createInstanceSamplingWithReplacement(partition, sampleSize_);
        }

        std::unique_ptr<IInstanceSampling> create(const BinaryCsrView& labelMatrix, BiPartition& partition,
                                                  IStatistics& statistics) const override {
            return createInstanceSamplingWithReplacement(partition, sampleSize_);
        }

        std::unique_ptr<IInstanceSampling> create(const CContiguousView<const float32>& regressionMatrix,
                                                  const SinglePartition& partition,
                                                  IStatistics& statistics) const override {
            return createInstanceSamplingWithReplacement(partition, sampleSize_);
        }

        std::unique_ptr<IInstanceSampling> create(const CContiguousView<const float32>& regressionMatrix,
                                                  BiPartition& partition, IStatistics& statistics) const override {
            return createInstanceSamplingWithReplacement(partition, sampleSize_);
        }

        std::unique_ptr<IInstanceSampling> create(const CsrView<const float32>& regressionMatrix,
                                                  const SinglePartition& partition,
                                                  IStatistics& statistics) const override {
            return createInstanceSamplingWithReplacement(partition, sampleSize_);
        }

        std::unique_ptr<IInstanceSampling> create(const CsrView<const float32>& regressionMatrix,
                                                  BiPartition& partition, IStatistics& statistics) const override {
            return createInstanceSamplingWithReplacement(partition, sampleSize_);
        }
};

InstanceSamplingWithReplacementConfig::InstanceSamplingWithReplacementConfig() : sampleSize_(0.66f) {}

float32 InstanceSamplingWithReplacementConfig::getSampleSize() const {
    return sampleSize_;
}

IInstanceSamplingWithReplacementConfig& InstanceSamplingWithReplacementConfig::setSampleSize(float32 sampleSize) {
    assertGreater<float32>("sampleSize", sampleSize, 0);
    assertLessOrEqual<float32>("sampleSize", sampleSize, 1);
    sampleSize_ = sampleSize;
    return *this;
}

std::unique_ptr<IClassificationInstanceSamplingFactory>
  InstanceSamplingWithReplacementConfig::createClassificationInstanceSamplingFactory() const {
    return std::make_unique<InstanceSamplingWithReplacementFactory>(sampleSize_);
}

std::unique_ptr<IRegressionInstanceSamplingFactory>
  InstanceSamplingWithReplacementConfig::createRegressionInstanceSamplingFactory() const {
    return std::make_unique<InstanceSamplingWithReplacementFactory>(sampleSize_);
}
