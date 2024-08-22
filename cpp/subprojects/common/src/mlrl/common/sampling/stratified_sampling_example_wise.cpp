#include "mlrl/common/sampling/stratified_sampling_example_wise.hpp"

#include "mlrl/common/sampling/partition_single.hpp"
#include "stratified_sampling_common.hpp"

#include <algorithm>
#include <unordered_map>

template<typename LabelMatrix, typename IndexIterator>
ExampleWiseStratification<LabelMatrix, IndexIterator>::ExampleWiseStratification(const LabelMatrix& labelMatrix,
                                                                                 IndexIterator indicesBegin,
                                                                                 IndexIterator indicesEnd)
    : numTotal_(indicesEnd - indicesBegin) {
    // Create a map that stores the indices of the examples that are associated with each unique label vector...
    typedef typename LabelMatrix::const_row Key;
    typedef typename LabelMatrix::const_row::Hash Hash;
    typedef typename LabelMatrix::const_row::Equal Equal;
    std::unordered_map<Key, std::unique_ptr<std::vector<uint32>>, Hash, Equal> map;

    for (uint32 i = 0; i < numTotal_; i++) {
        uint32 exampleIndex = indicesBegin[i];
        std::unique_ptr<std::vector<uint32>>& exampleIndicesPtr = map[labelMatrix[exampleIndex]];

        if (!exampleIndicesPtr) {
            exampleIndicesPtr = std::make_unique<std::vector<uint32>>();
        }

        exampleIndicesPtr->emplace_back(exampleIndex);
    }

    // Sort the label vectors by their frequency...
    order_.reserve(map.size());

    for (auto it = map.begin(); it != map.end(); it++) {
        auto& entry = *it;
        std::unique_ptr<std::vector<uint32>>& exampleIndicesPtr = entry.second;
        order_.emplace_back(std::move(exampleIndicesPtr));
    }

    std::sort(order_.begin(), order_.end(),
              [=](const std::unique_ptr<std::vector<uint32>>& lhs, const std::unique_ptr<std::vector<uint32>>& rhs) {
        return lhs->size() < rhs->size();
    });
}

template<typename LabelMatrix, typename IndexIterator>
void ExampleWiseStratification<LabelMatrix, IndexIterator>::sampleWeights(BitWeightVector& weightVector,
                                                                          float32 sampleSize, RNG& rng) const {
    uint32 numTotalSamples = static_cast<uint32>(std::round(sampleSize * numTotal_));
    uint32 numTotalOutOfSamples = numTotal_ - numTotalSamples;
    uint32 numNonZeroWeights = 0;
    uint32 numZeroWeights = 0;

    for (auto it = order_.begin(); it != order_.end(); it++) {
        const std::unique_ptr<std::vector<uint32>>& exampleIndicesPtr = *it;
        std::vector<uint32>::iterator indexIterator = exampleIndicesPtr->begin();
        uint32 numExamples = exampleIndicesPtr->size();
        float32 numSamplesDecimal = sampleSize * numExamples;
        uint32 numDesiredSamples = numTotalSamples - numNonZeroWeights;
        uint32 numDesiredOutOfSamples = numTotalOutOfSamples - numZeroWeights;
        uint32 numSamples =
          static_cast<uint32>(tiebreak(numDesiredSamples, numDesiredOutOfSamples, rng) ? std::ceil(numSamplesDecimal)
                                                                                       : std::floor(numSamplesDecimal));
        numNonZeroWeights += numSamples;
        numZeroWeights += (numExamples - numSamples);

        // Use the Fisher-Yates shuffle to randomly draw `numSamples` examples and set their weight to 1...
        uint32 i;

        for (i = 0; i < numSamples; i++) {
            uint32 randomIndex = rng.randomInt(i, numExamples);
            uint32 exampleIndex = indexIterator[randomIndex];
            indexIterator[randomIndex] = indexIterator[i];
            indexIterator[i] = exampleIndex;
            weightVector.set(exampleIndex, true);
        }

        // Set the weights of the remaining examples to 0...
        for (; i < numExamples; i++) {
            uint32 exampleIndex = indexIterator[i];
            weightVector.set(exampleIndex, false);
        }
    }

    weightVector.setNumNonZeroWeights(numNonZeroWeights);
}

template<typename LabelMatrix, typename IndexIterator>
void ExampleWiseStratification<LabelMatrix, IndexIterator>::sampleBiPartition(BiPartition& partition, RNG& rng) const {
    BiPartition::iterator firstIterator = partition.first_begin();
    BiPartition::iterator secondIterator = partition.second_begin();
    uint32 numFirst = partition.getNumFirst();
    uint32 numSecond = partition.getNumSecond();

    for (auto it = order_.begin(); it != order_.end(); it++) {
        const std::unique_ptr<std::vector<uint32>>& exampleIndicesPtr = *it;
        std::vector<uint32>::iterator indexIterator = exampleIndicesPtr->begin();
        uint32 numExamples = exampleIndicesPtr->size();
        float32 sampleSize = (float32) numFirst / (float32) (numFirst + numSecond);
        float32 numSamplesDecimal = sampleSize * numExamples;
        uint32 numSamples = static_cast<uint32>(tiebreak(numFirst, numSecond, rng) ? std::ceil(numSamplesDecimal)
                                                                                   : std::floor(numSamplesDecimal));

        // Ensure that we do not add too many examples to the first or second partition...
        if (numSamples > numFirst) {
            numSamples = numFirst;
        } else if (numExamples - numSamples > numSecond) {
            numSamples = numExamples - numSecond;
        }

        numFirst -= numSamples;
        numSecond -= (numExamples - numSamples);

        // Use the Fisher-Yates shuffle to randomly draw `numSamples` examples and add them to the first set...
        uint32 i;

        for (i = 0; i < numSamples; i++) {
            uint32 randomIndex = rng.randomInt(i, numExamples);
            uint32 exampleIndex = indexIterator[randomIndex];
            indexIterator[randomIndex] = indexIterator[i];
            indexIterator[i] = exampleIndex;
            *firstIterator = exampleIndex;
            firstIterator++;
        }

        // Add the remaining examples to the second set...
        for (; i < numExamples; i++) {
            uint32 exampleIndex = indexIterator[i];
            *secondIterator = exampleIndex;
            secondIterator++;
        }
    }
}

template class ExampleWiseStratification<CContiguousView<const uint8>, SinglePartition::const_iterator>;
template class ExampleWiseStratification<CContiguousView<const uint8>, BiPartition::const_iterator>;
template class ExampleWiseStratification<BinaryCsrView, SinglePartition::const_iterator>;
template class ExampleWiseStratification<BinaryCsrView, BiPartition::const_iterator>;
