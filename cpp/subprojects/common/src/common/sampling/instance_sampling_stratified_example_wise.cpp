#include "common/sampling/instance_sampling_stratified_example_wise.hpp"
#include "common/sampling/weight_vector_dense.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/sampling/partition_single.hpp"
#include "stratified_sampling.hpp"
#include <unordered_map>
#include <algorithm>
#include <vector>
#include <cmath>


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

        uint32 numTrainingExamples_;

        float32 sampleSize_;

        DenseWeightVector<uint8> weightVector_;

        typedef typename LabelMatrix::view_type Key;

        typedef typename LabelMatrix::view_type::Hash Hash;

        typedef typename LabelMatrix::view_type::Pred Pred;

        std::unordered_map<Key, std::vector<uint32>, Hash, Pred> map_;

        std::vector<std::reference_wrapper<std::vector<uint32>>> order_;

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
            : numTrainingExamples_(indicesEnd - indicesBegin), sampleSize_(sampleSize),
              weightVector_(DenseWeightVector<uint8>(labelMatrix.getNumRows())) {
            // Create a map that stores the indices of the examples that are associated with each unique label vector...
            for (uint32 i = 0; i < numTrainingExamples_; i++) {
                uint32 exampleIndex = indicesBegin[i];
                std::vector<uint32>& exampleIndices = map_[labelMatrix.createView(exampleIndex)];
                exampleIndices.push_back(exampleIndex);
            }

            // Sort the label vectors by their frequency...
            order_.reserve(map_.size());

            for (auto it = map_.begin(); it != map_.end(); it++) {
                auto& entry = *it;
                std::vector<uint32>& exampleIndices = entry.second;
                order_.push_back(exampleIndices);
            }

            std::sort(order_.begin(), order_.end(), [=](const std::vector<uint32>& a, const std::vector<uint32>& b) {
                return a.size() < b.size();
            });
        }

        const IWeightVector& subSample(RNG& rng) override {
            DenseWeightVector<uint8>::iterator weightIterator = weightVector_.begin();
            uint32 numTotalSamples = (uint32) std::round(sampleSize_ * numTrainingExamples_);
            uint32 numTotalOutOfSamples = numTrainingExamples_ - numTotalSamples;
            uint32 numNonZeroWeights = 0;
            uint32 numZeroWeights = 0;

            for (auto it = order_.begin(); it != order_.end(); it++) {
                std::vector<uint32>& exampleIndices = *it;
                std::vector<uint32>::iterator indexIterator = exampleIndices.begin();
                uint32 numExamples = exampleIndices.size();
                float32 numSamplesDecimal = sampleSize_ * numExamples;
                uint32 numDesiredSamples = numTotalSamples - numNonZeroWeights;
                uint32 numDesiredOutOfSamples = numTotalOutOfSamples - numZeroWeights;
                uint32 numSamples = (uint32) (tiebreak(numDesiredSamples, numDesiredOutOfSamples, rng) ? 
                                              std::ceil(numSamplesDecimal) : std::floor(numSamplesDecimal));
                numNonZeroWeights += numSamples;
                numZeroWeights += (numExamples - numSamples);

                // Use the Fisher-Yates shuffle to randomly draw `numSamples` examples and set their weight to 1...
                uint32 i;

                for (i = 0; i < numSamples; i++) {
                    uint32 randomIndex = rng.random(i, numExamples);
                    uint32 exampleIndex = indexIterator[randomIndex];
                    indexIterator[randomIndex] = indexIterator[i];
                    indexIterator[i] = exampleIndex;
                    weightIterator[exampleIndex] = 1;
                }

                // Set the weights of the remaining examples to 0...
                for (i = i + 1; i < numExamples; i++) {
                    uint32 exampleIndex = indexIterator[i];
                    weightIterator[exampleIndex] = 0;
                }
            }

            weightVector_.setNumNonZeroWeights(numNonZeroWeights);
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
