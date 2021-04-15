#include "common/sampling/instance_sampling_random.hpp"
#include "common/indices/index_iterator.hpp"
#include "weight_sampling.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/sampling/partition_single.hpp"


static inline std::unique_ptr<IWeightVector> subSampleInternally(const SinglePartition& partition, float32 sampleSize,
                                                                 RNG& rng) {
    uint32 numExamples = partition.getNumElements();
    uint32 numSamples = (uint32) (sampleSize * numExamples);
    return sampleWeightsWithoutReplacement<IndexIterator>(IndexIterator(numExamples), numExamples, numSamples,
                                                          numExamples, rng);
}

static inline std::unique_ptr<IWeightVector> subSampleInternally(BiPartition& partition, float32 sampleSize, RNG& rng) {
    uint32 numExamples = partition.getNumElements();
    uint32 numTrainingExamples = partition.getNumFirst();
    uint32 numSamples = (uint32) (sampleSize * numTrainingExamples);
    return sampleWeightsWithoutReplacement<BiPartition::const_iterator>(partition.first_cbegin(), numTrainingExamples,
                                                                        numSamples, numExamples, rng);
}

/**
 * Allows to select a subset of the available training examples without replacement.
 *
 * @tparam Partition The type of the object that provides access to the indices of the examples that are included in the
 *                   training set
 */
template<class Partition>
class RandomInstanceSubsetSelection final : public IInstanceSubSampling {

    private:

        Partition& partition_;

        float32 sampleSize_;

    public:

        /**
         * @param partition  A reference to an object of template type `Partition` that provides access to the indices
         *                   of the examples that are included in the training set
         * @param sampleSize The fraction of examples to be included in the sample (e.g. a value of 0.6 corresponds to
         *                   60 % of the available examples). Must be in (0, 1)
         */
        RandomInstanceSubsetSelection(Partition& partition, float32 sampleSize)
            : partition_(partition), sampleSize_(sampleSize) {

        }

        std::unique_ptr<IWeightVector> subSample(RNG& rng) override {
            return subSampleInternally(partition_, sampleSize_, rng);
        }

};

RandomInstanceSubsetSelectionFactory::RandomInstanceSubsetSelectionFactory(float32 sampleSize)
    : sampleSize_(sampleSize) {

}

std::unique_ptr<IInstanceSubSampling> RandomInstanceSubsetSelectionFactory::create(
        const CContiguousLabelMatrix& labelMatrix, const SinglePartition& partition) const {
    return std::make_unique<RandomInstanceSubsetSelection<const SinglePartition>>(partition, sampleSize_);
}

std::unique_ptr<IInstanceSubSampling> RandomInstanceSubsetSelectionFactory::create(
        const CContiguousLabelMatrix& labelMatrix, BiPartition& partition) const {
    return std::make_unique<RandomInstanceSubsetSelection<BiPartition>>(partition, sampleSize_);
}

std::unique_ptr<IInstanceSubSampling> RandomInstanceSubsetSelectionFactory::create(
        const CsrLabelMatrix& labelMatrix, const SinglePartition& partition) const {
    return std::make_unique<RandomInstanceSubsetSelection<const SinglePartition>>(partition, sampleSize_);
}

std::unique_ptr<IInstanceSubSampling> RandomInstanceSubsetSelectionFactory::create(
        const CsrLabelMatrix& labelMatrix, BiPartition& partition) const {
    return std::make_unique<RandomInstanceSubsetSelection<BiPartition>>(partition, sampleSize_);
}
