#include "common/sampling/instance_sampling_random.hpp"
#include "common/indices/index_iterator.hpp"
#include "weight_sampling.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/sampling/partition_single.hpp"


/**
 * Allows to select a subset of the available training examples without replacement.
 */
class RandomInstanceSubsetSelection final : public IInstanceSubSampling {

    private:

        float32 sampleSize_;

    public:

        /**
         * @param sampleSize The fraction of examples to be included in the sample (e.g. a value of 0.6 corresponds to
         *                   60 % of the available examples). Must be in (0, 1)
         */
        RandomInstanceSubsetSelection(float32 sampleSize)
            : sampleSize_(sampleSize) {

        }

        std::unique_ptr<IWeightVector> subSample(const SinglePartition& partition, RNG& rng) const override {
            uint32 numExamples = partition.getNumElements();
            uint32 numSamples = (uint32) (sampleSize_ * numExamples);
            return sampleWeightsWithoutReplacement<IndexIterator>(IndexIterator(numExamples), numExamples, numSamples,
                                                                  numExamples, rng);
        }

        std::unique_ptr<IWeightVector> subSample(const BiPartition& partition, RNG& rng) const override {
            uint32 numExamples = partition.getNumElements();
            uint32 numTrainingExamples = partition.getNumFirst();
            uint32 numSamples = (uint32) (sampleSize_ * numTrainingExamples);
            return sampleWeightsWithoutReplacement<BiPartition::const_iterator>(partition.first_cbegin(),
                                                                                numTrainingExamples, numSamples,
                                                                                numExamples, rng);
        }

};

RandomInstanceSubsetSelectionFactory::RandomInstanceSubsetSelectionFactory(float32 sampleSize)
    : sampleSize_(sampleSize) {

}

std::unique_ptr<IInstanceSubSampling> RandomInstanceSubsetSelectionFactory::create(
        const CContiguousLabelMatrix& labelMatrix) const {
    return std::make_unique<RandomInstanceSubsetSelection>(sampleSize_);
}

std::unique_ptr<IInstanceSubSampling> RandomInstanceSubsetSelectionFactory::create(
        const CsrLabelMatrix& labelMatrix) const {
    return std::make_unique<RandomInstanceSubsetSelection>(sampleSize_);
}
