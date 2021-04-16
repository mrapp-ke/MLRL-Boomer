#include "common/sampling/instance_sampling_stratified_example_wise.hpp"
#include "common/sampling/weight_vector_dense.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/sampling/partition_single.hpp"
#include "common/input/label_vector_set.hpp"
#include <vector>
#include <cmath>


static inline bool tiebreak(uint32 numDesiredSamples, uint32 numDesiredOutOfSamples, RNG& rng) {
    if (numDesiredSamples > numDesiredOutOfSamples) {
        return true;
    } else if (numDesiredSamples < numDesiredOutOfSamples) {
        return false;
    } else {
        return rng.random(0, 2) != 0;
    }
}

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

        uint32 numTotalExamples_;

        uint32 numTrainingExamples_;

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
            : numTotalExamples_(labelMatrix.getNumRows()), numTrainingExamples_(indicesEnd - indicesBegin), 
              sampleSize_(sampleSize) {
            for (uint32 i = 0; i < numTrainingExamples_; i++) {
                uint32 exampleIndex = indicesBegin[i];
                std::vector<uint32>& exampleIndices =
                    labelVectors_.addLabelVector(labelMatrix.getLabelVector(exampleIndex));
                exampleIndices.push_back(exampleIndex);
            }
        }

        std::unique_ptr<IWeightVector> subSample(RNG& rng) override {
            // Create a vector to store the weights of individual examples...
            std::unique_ptr<DenseWeightVector<uint8>> weightVectorPtr =
                std::make_unique<DenseWeightVector<uint8>>(numTotalExamples_, true);
            DenseWeightVector<uint8>::iterator weightIterator = weightVectorPtr->begin();

            // For each label vector, sample some of the examples with these labels...
            uint32 numTotalSamples = (uint32) std::round(sampleSize_ * numTrainingExamples_);
            uint32 numTotalOutOfSamples = numTrainingExamples_ - numTotalSamples;
            uint32 numNonZeroWeights = 0;
            uint32 numZeroWeights = 0;

            for (auto it = labelVectors_.cbegin(); it != labelVectors_.cend(); it++) {
                const auto& entry = *it;
                std::vector<uint32> exampleIndices = entry.second;
                std::vector<uint32>::iterator indexIterator = exampleIndices.begin();
                std::vector<uint32>::size_type numExamples = exampleIndices.size();
                float32 numSamplesDecimal = sampleSize_ * numExamples;
                uint32 numDesiredSamples = numTotalSamples - numNonZeroWeights;
                uint32 numDesiredOutOfSamples = numTotalOutOfSamples - numZeroWeights;
                uint32 numSamples = (uint32) (tiebreak(numDesiredSamples, numDesiredOutOfSamples, rng) ? 
                                              std::ceil(numSamplesDecimal) : std::floor(numSamplesDecimal));
                numNonZeroWeights += numSamples;
                numZeroWeights += (numExamples - numSamples);

                // Use the Fisher-Yates shuffle to randomly draw `numSamples` examples and set their weight to 1...
                for (uint32 i = 0; i < numSamples; i++) {
                    uint32 randomIndex = rng.random(i, numExamples);
                    uint32 exampleIndex = indexIterator[randomIndex];
                    indexIterator[randomIndex] = indexIterator[i];
                    indexIterator[i] = exampleIndex;
                    weightIterator[exampleIndex] = 1;
                }
            }

            weightVectorPtr->setNumNonZeroWeights(numNonZeroWeights);
            return weightVectorPtr;
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
