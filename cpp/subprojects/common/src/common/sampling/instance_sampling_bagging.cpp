#include "common/sampling/instance_sampling_bagging.hpp"
#include "common/sampling/weight_vector_dense.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/sampling/partition_single.hpp"
#include "common/data/arrays.hpp"


static inline void subSampleInternally(const SinglePartition& partition, float32 sampleSize,
                                       DenseWeightVector<uint32>& weightVector, RNG& rng) {
    uint32 numExamples = partition.getNumElements();
    uint32 numSamples = (uint32) (sampleSize * numExamples);
    typename DenseWeightVector<uint32>::iterator weightIterator = weightVector.begin();
    setArrayToZeros(weightIterator, numExamples);
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

static inline void subSampleInternally(BiPartition& partition, float32 sampleSize,
                                       DenseWeightVector<uint32>& weightVector, RNG& rng) {
    uint32 numExamples = partition.getNumElements();
    uint32 numTrainingExamples = partition.getNumFirst();
    uint32 numSamples = (uint32) (sampleSize * numTrainingExamples);
    BiPartition::const_iterator indexIterator = partition.first_cbegin();
    typename DenseWeightVector<uint32>::iterator weightIterator = weightVector.begin();
    setArrayToZeros(weightIterator, numExamples);
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
 * Implements bootstrap aggregating (bagging) for selecting a subset of the available training examples with
 * replacement.
 *
 * @tparam Partition The type of the object that provides access to the indices of the examples that are included in the
 *                   training set
 */
template<typename Partition>
class Bagging final : public IInstanceSampling {

    private:

        Partition& partition_;

        float32 sampleSize_;

        DenseWeightVector<uint32> weightVector_;

    public:

        /**
         * @param partition  A reference to an object of template type `Partition` that provides access to the indices
         *                   of the examples that are included in the training set
         * @param sampleSize The fraction of examples to be included in the sample (e.g. a value of 0.6 corresponds to
         *                   60 % of the available examples). Must be in (0, 1]
         */
        Bagging(Partition& partition, float32 sampleSize)
            : partition_(partition), sampleSize_(sampleSize),
              weightVector_(DenseWeightVector<uint32>(partition.getNumElements())) {

        }

        const IWeightVector& subSample(RNG& rng) override {
            subSampleInternally(partition_, sampleSize_, weightVector_, rng);
            return weightVector_;
        }

};

BaggingFactory::BaggingFactory(float32 sampleSize)
    : sampleSize_(sampleSize) {

}

std::unique_ptr<IInstanceSampling> BaggingFactory::create(const CContiguousLabelMatrix& labelMatrix,
                                                          const SinglePartition& partition,
                                                          IStatistics& statistics) const {
    return std::make_unique<Bagging<const SinglePartition>>(partition, sampleSize_);
}

std::unique_ptr<IInstanceSampling> BaggingFactory::create(const CContiguousLabelMatrix& labelMatrix,
                                                          BiPartition& partition, IStatistics& statistics) const {
    return std::make_unique<Bagging<BiPartition>>(partition, sampleSize_);
}

std::unique_ptr<IInstanceSampling> BaggingFactory::create(const CsrLabelMatrix& labelMatrix,
                                                          const SinglePartition& partition,
                                                          IStatistics& statistics) const {
    return std::make_unique<Bagging<const SinglePartition>>(partition, sampleSize_);
}

std::unique_ptr<IInstanceSampling> BaggingFactory::create(const CsrLabelMatrix& labelMatrix, BiPartition& partition,
                                                          IStatistics& statistics) const {
    return std::make_unique<Bagging<BiPartition>>(partition, sampleSize_);
}
