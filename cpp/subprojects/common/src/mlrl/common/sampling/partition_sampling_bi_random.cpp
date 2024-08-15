#include "mlrl/common/sampling/partition_sampling_bi_random.hpp"

#include "index_sampling.hpp"
#include "mlrl/common/sampling/partition_bi.hpp"
#include "mlrl/common/util/validation.hpp"

/**
 * Allows to randomly split the training examples into two mutually exclusive sets that may be used as a training set
 * and a holdout set.
 */
class RandomBiPartitionSampling final : public IPartitionSampling {
    private:

        BiPartition partition_;

    public:

        /**
         * @param numTraining   The number of examples to be included in the training set
         * @param numHoldout    The number of examples to be included in the holdout set
         */
        RandomBiPartitionSampling(uint32 numTraining, uint32 numHoldout) : partition_(numTraining, numHoldout) {}

        IPartition& partition(RNG& rng) override {
            uint32 numTraining = partition_.getNumFirst();
            uint32 numHoldout = partition_.getNumSecond();
            BiPartition::iterator trainingIterator = partition_.first_begin();
            util::setViewToIncreasingValues(trainingIterator, numTraining, 0, 1);
            BiPartition::iterator holdoutIterator = partition_.second_begin();

            for (uint32 i = 0; i < numHoldout; i++) {
                holdoutIterator[i] = numTraining + i;
            }

            uint32 numTotal = partition_.getNumElements();
            randomPermutation<BiPartition::iterator, BiPartition::iterator>(trainingIterator, holdoutIterator,
                                                                            numTraining, numTotal, numTraining, rng);
            return partition_;
        }
};

template<typename OutputMatrix>
static inline std::unique_ptr<IPartitionSampling> createRandomBiPartitionSampling(const OutputMatrix& outputMatrix,
                                                                                  float32 holdoutSetSize) {
    uint32 numExamples = outputMatrix.numRows;
    uint32 numHoldout = static_cast<uint32>(holdoutSetSize * numExamples);
    uint32 numTraining = numExamples - numHoldout;
    return std::make_unique<RandomBiPartitionSampling>(numTraining, numHoldout);
}

/**
 * Allows to create objects of the type `IPartitionSampling` that randomly split the training examples into two mutually
 * exclusive sets that may be used as a training set and a holdout set.
 */
class RandomBiPartitionSamplingFactory final : public IClassificationPartitionSamplingFactory,
                                               public IRegressionPartitionSamplingFactory {
    private:

        const float32 holdoutSetSize_;

    public:

        /**
         * @param holdoutSetSize The fraction of examples to be included in the holdout set (e.g. a value of 0.6
         *                       corresponds to 60 % of the available examples). Must be in (0, 1)
         */
        RandomBiPartitionSamplingFactory(float32 holdoutSetSize) : holdoutSetSize_(holdoutSetSize) {}

        std::unique_ptr<IPartitionSampling> create(const CContiguousView<const uint8>& labelMatrix) const override {
            return createRandomBiPartitionSampling(labelMatrix, holdoutSetSize_);
        }

        std::unique_ptr<IPartitionSampling> create(const BinaryCsrView& labelMatrix) const override {
            return createRandomBiPartitionSampling(labelMatrix, holdoutSetSize_);
        }

        std::unique_ptr<IPartitionSampling> create(
          const CContiguousView<const float32>& regressionMatrix) const override {
            return createRandomBiPartitionSampling(regressionMatrix, holdoutSetSize_);
        }

        std::unique_ptr<IPartitionSampling> create(const CsrView<const float32>& regressionMatrix) const override {
            return createRandomBiPartitionSampling(regressionMatrix, holdoutSetSize_);
        }
};

RandomBiPartitionSamplingConfig::RandomBiPartitionSamplingConfig() : holdoutSetSize_(0.33f) {}

float32 RandomBiPartitionSamplingConfig::getHoldoutSetSize() const {
    return holdoutSetSize_;
}

IRandomBiPartitionSamplingConfig& RandomBiPartitionSamplingConfig::setHoldoutSetSize(float32 holdoutSetSize) {
    util::assertGreater<float32>("holdoutSetSize", holdoutSetSize, 0);
    util::assertLess<float32>("holdoutSetSize", holdoutSetSize, 1);
    holdoutSetSize_ = holdoutSetSize;
    return *this;
}

std::unique_ptr<IClassificationPartitionSamplingFactory>
  RandomBiPartitionSamplingConfig::createClassificationPartitionSamplingFactory() const {
    return std::make_unique<RandomBiPartitionSamplingFactory>(holdoutSetSize_);
}

std::unique_ptr<IRegressionPartitionSamplingFactory>
  RandomBiPartitionSamplingConfig::createRegressionPartitionSamplingFactory() const {
    return std::make_unique<RandomBiPartitionSamplingFactory>(holdoutSetSize_);
}
