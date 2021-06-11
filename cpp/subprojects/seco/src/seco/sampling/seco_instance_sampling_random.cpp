#include <common/statistics/statistics.hpp>
#include <common/statistics/statistics_provider.hpp>
#include <vector>
#include "seco/sampling/seco_instance_sampling_random.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/sampling/partition_single.hpp"
#include "common/sampling/weight_vector_dense.hpp"

#include "../../../../common/src/common/sampling/weight_sampling.hpp"
#include "../statistics/statistics_label_wise_common.hpp"


template <typename Iterator>
static inline void subSampleInternally(Iterator iterator, float32 sampleSize, uint32 numExamples,
                                       BitWeightVector& weightVector, RNG& rng) {
    auto numSamples = (uint32) (sampleSize * numExamples);
    sampleWeightsWithoutReplacement<BiPartition::const_iterator>(weightVector,  iterator, numExamples,
                                                                 numSamples, rng);
}

/**
 * Allows to select a subset of the available training examples without replacement.
 *
 * @tparam Partition The type of the object that provides access to the indices of the examples that are included in the
 *                   training set
 */
template<class Partition>
class SecoRandomInstanceSubsetSelection final : public IInstanceSubSampling {

private:

    Partition& partition_;

    float32 sampleSize_;

    BitWeightVector weightVector_;

    unsigned int* exampleCoverage_;

    uint32 size_;

public:

    /**
     * @param partition  A reference to an object of template type `Partition` that provides access to the indices
     *                   of the examples that are included in the training set
     * @param sampleSize The fraction of examples to be included in the sample (e.g. a value of 0.6 corresponds to
     *                   60 % of the available examples). Must be in (0, 1)
     */
    SecoRandomInstanceSubsetSelection(Partition& partition, float32 sampleSize)
            : partition_(partition), sampleSize_(sampleSize),
              weightVector_(BitWeightVector(partition.getNumElements())) {
        exampleCoverage_ = new unsigned int[partition.getNumElements()];
        size_ = 0;
    }

    ~SecoRandomInstanceSubsetSelection() override {
        delete[] exampleCoverage_;
    }

    const IWeightVector& subSample(RNG& rng) override {
        subSampleInternally(exampleCoverage_, sampleSize_, size_, weightVector_, rng);
        return weightVector_;
    }

    void setWeights(IStatistics& statistics) override {
        seco::DenseWeightMatrix* weights = dynamic_cast<seco::ICoverageStatistics&>(statistics).getUncoveredWeights();
        int i = 0;
        bool unCovered = false;

        for (uint32 row = 0; row < weights->getNumRows(); row++, unCovered = false) {
            seco::DenseWeightMatrix::CContiguousConstView::const_iterator it = weights->row_cbegin(row);

            for (uint32 column = 0; column < weights->getNumCols(); column++) {
                float64 weight = it[column];
                if (weight > 0) {
                    // example is unCovered
                    unCovered = true;
                    break;
                }
            }
            // if example is unCovered it is added to the array
            if (unCovered) {
                exampleCoverage_[i++] = row;
            }
        }
        size_ = i;
    }

};

SecoRandomInstanceSubsetSelectionFactory::SecoRandomInstanceSubsetSelectionFactory(float32 sampleSize)
        : sampleSize_(sampleSize) {
}

std::unique_ptr<IInstanceSubSampling> SecoRandomInstanceSubsetSelectionFactory::create(
        const CContiguousLabelMatrix& labelMatrix, const SinglePartition& partition, IStatistics& statistics) const {
    return std::make_unique<SecoRandomInstanceSubsetSelection<const SinglePartition>>(partition, sampleSize_);
}

std::unique_ptr<IInstanceSubSampling> SecoRandomInstanceSubsetSelectionFactory::create(
        const CContiguousLabelMatrix& labelMatrix, BiPartition& partition, IStatistics& statistics) const {
    return std::make_unique<SecoRandomInstanceSubsetSelection<BiPartition>>(partition, sampleSize_);
}

std::unique_ptr<IInstanceSubSampling> SecoRandomInstanceSubsetSelectionFactory::create(
        const CsrLabelMatrix& labelMatrix, const SinglePartition& partition, IStatistics& statistics) const {
    return std::make_unique<SecoRandomInstanceSubsetSelection<const SinglePartition>>(partition, sampleSize_);
}

std::unique_ptr<IInstanceSubSampling> SecoRandomInstanceSubsetSelectionFactory::create(
        const CsrLabelMatrix& labelMatrix, BiPartition& partition, IStatistics& statistics) const {
    return std::make_unique<SecoRandomInstanceSubsetSelection<BiPartition>>(partition, sampleSize_);
}
