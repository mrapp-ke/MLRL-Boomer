#include <common/statistics/statistics.hpp>
#include <common/statistics/statistics_provider.hpp>
#include "seco/sampling/instance_sampling_random.hpp"
#include "common/indices/index_iterator.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/sampling/partition_single.hpp"
#include "common/sampling/weight_vector_dense.hpp"

#include "../../../../common/src/common/sampling/weight_sampling.hpp"
#include "../statistics/statistics_label_wise_common.hpp"


static inline void subSampleInternally(const SinglePartition& partition, float32 sampleSize,
                                       DenseWeightVector<uint8>& weightVector, RNG& rng) {
    uint32 numExamples = partition.getNumElements();
    uint32 numSamples = (uint32) (sampleSize * numExamples);
    sampleWeightsWithoutReplacement<IndexIterator>(weightVector, IndexIterator(numExamples), numExamples, numSamples,
                                                   rng);
}

static inline void subSampleInternally(BiPartition& partition, float32 sampleSize,
                                       DenseWeightVector<uint8>& weightVector, RNG& rng) {
    uint32 numTrainingExamples = partition.getNumFirst();
    uint32 numSamples = (uint32) (sampleSize * numTrainingExamples);
    sampleWeightsWithoutReplacement<BiPartition::const_iterator>(weightVector, partition.first_cbegin(),
                                                                 numTrainingExamples, numSamples, rng);
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

    DenseWeightVector<uint8> weightVector_;

public:

    /**
     * @param partition  A reference to an object of template type `Partition` that provides access to the indices
     *                   of the examples that are included in the training set
     * @param sampleSize The fraction of examples to be included in the sample (e.g. a value of 0.6 corresponds to
     *                   60 % of the available examples). Must be in (0, 1)
     */
    RandomInstanceSubsetSelection(Partition& partition, float32 sampleSize)
            : partition_(partition), sampleSize_(sampleSize),
              weightVector_(DenseWeightVector<uint8>(partition.getNumElements())) {

    }

    const IWeightVector& subSample(RNG& rng) override {
        subSampleInternally(partition_, sampleSize_, weightVector_, rng);
        return weightVector_;
    }

    const void setWeights(IStatistics& statistics) override {
        seco::DenseWeightMatrix* weights = dynamic_cast<seco::ICoverageStatistics&>(statistics).getUncoveredWeights();

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
