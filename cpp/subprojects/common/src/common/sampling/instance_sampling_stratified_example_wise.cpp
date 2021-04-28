#include "common/sampling/instance_sampling_stratified_example_wise.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/sampling/partition_single.hpp"
#include "stratified_sampling.hpp"


/**
 * Implements stratified sampling, where distinct label vectors are treated as individual classes.
 *
 * @tparam LabelMatrix      The type of the label matrix that provides random or row-wise access to the labels of the
 *                          training examples
 * @tparam IndexIterator    The type of the iterator that provides access to the indices of the examples that are
 *                          contained by the training set
 */
template<class LabelMatrix, class IndexIterator>
class ExampleWiseStratifiedSampling final : public IInstanceSubSampling {

    private:

        float32 sampleSize_;

        DenseWeightVector<uint8> weightVector_;

        ExampleWiseStratification<LabelMatrix, IndexIterator> stratification_;

    public:

        /**
         * @param labelMatrix   A reference to an object of template type `LabelMatrix` that provides random or row-wise
         *                      access to the labels of the training examples
         * @param indicesBegin  An iterator to the beginning of the indices of the examples that are contained by the
         *                      training set
         * @param indicesEnd    An iterator to the end of the indices of the examples that are contained by the training
         *                      set
         * @param sampleSize    The fraction of examples to be included in the sample (e.g. a value of 0.6 corresponds
         *                      to 60 % of the available examples). Must be in (0, 1]
         */
        ExampleWiseStratifiedSampling(const LabelMatrix& labelMatrix, IndexIterator indicesBegin,
                                      IndexIterator indicesEnd, float32 sampleSize)
            : sampleSize_(sampleSize), weightVector_(DenseWeightVector<uint8>(labelMatrix.getNumRows(),
                                                     (uint32) (indicesEnd - indicesBegin) < labelMatrix.getNumRows())),
              stratification_(ExampleWiseStratification<LabelMatrix, IndexIterator>(labelMatrix, indicesBegin,
                                                                                    indicesEnd)) {

        }

        const IWeightVector& subSample(RNG& rng) override {
            stratification_.sampleWeights(weightVector_, sampleSize_, rng);
            return weightVector_;
        }

};

ExampleWiseStratifiedSamplingFactory::ExampleWiseStratifiedSamplingFactory(float32 sampleSize)
    : sampleSize_(sampleSize) {

}

std::unique_ptr<IInstanceSubSampling> ExampleWiseStratifiedSamplingFactory::create(
        const CContiguousLabelMatrix& labelMatrix, const SinglePartition& partition) const {
    return std::make_unique<ExampleWiseStratifiedSampling<CContiguousLabelMatrix, SinglePartition::const_iterator>>(
        labelMatrix, partition.cbegin(), partition.cend(), sampleSize_);
}

std::unique_ptr<IInstanceSubSampling> ExampleWiseStratifiedSamplingFactory::create(
        const CContiguousLabelMatrix& labelMatrix, BiPartition& partition) const {
    return std::make_unique<ExampleWiseStratifiedSampling<CContiguousLabelMatrix, BiPartition::const_iterator>>(
        labelMatrix, partition.first_cbegin(), partition.first_cend(), sampleSize_);
}

std::unique_ptr<IInstanceSubSampling> ExampleWiseStratifiedSamplingFactory::create(
        const CsrLabelMatrix& labelMatrix, const SinglePartition& partition) const {
    return std::make_unique<ExampleWiseStratifiedSampling<CsrLabelMatrix, SinglePartition::const_iterator>>(
        labelMatrix, partition.cbegin(), partition.cend(), sampleSize_);
}

std::unique_ptr<IInstanceSubSampling> ExampleWiseStratifiedSamplingFactory::create(
        const CsrLabelMatrix& labelMatrix, BiPartition& partition) const {
    return std::make_unique<ExampleWiseStratifiedSampling<CsrLabelMatrix, BiPartition::const_iterator>>(
        labelMatrix, partition.first_cbegin(), partition.first_cend(), sampleSize_);
}
