#include "common/sampling/instance_sampling_stratified_example_wise.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/sampling/partition_single.hpp"
#include "common/input/label_vector_set.hpp"
#include <vector>


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

        LabelVectorSet<std::vector<uint32>> labelVectors_;

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
            : sampleSize_(sampleSize) {
            uint32 numExamples = indicesEnd - indicesBegin;

            for (uint32 i = 0; i < numExamples; i++) {
                uint32 exampleIndex = indicesBegin[i];
                std::vector<uint32>& exampleIndices =
                    labelVectors_.addLabelVector(labelMatrix.getLabelVector(exampleIndex));
                exampleIndices.push_back(exampleIndex);
            }
        }

        std::unique_ptr<IWeightVector> subSample(RNG& rng) override {
            // TODO Implement
            return nullptr;
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
