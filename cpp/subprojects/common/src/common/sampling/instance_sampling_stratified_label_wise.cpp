#include "common/sampling/instance_sampling_stratified_label_wise.hpp"
#include "common/sampling/weight_vector_dense.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/sampling/partition_single.hpp"
#include "common/input/label_matrix_csc.hpp"
#include "stratified_sampling.hpp"
#include <unordered_map>
#include <map>
#include <cstdlib>
#include <cmath>


static inline void updateNumExamplesPerLabel(const CContiguousLabelMatrix& labelMatrix, uint32 exampleIndex,
                                             uint32* numExamplesPerLabel,
                                             std::unordered_map<uint32, uint32>& affectedLabelIndices) {
    CContiguousLabelMatrix::value_const_iterator labelIterator = labelMatrix.row_values_cbegin(exampleIndex);
    uint32 numLabels = labelMatrix.getNumCols();

    for (uint32 i = 0; i < numLabels; i++) {
        if (labelIterator[i]) {
            uint32 numRemaining = numExamplesPerLabel[i];
            numExamplesPerLabel[i] = numRemaining - 1;
            affectedLabelIndices.emplace(i, numRemaining);
        }
    }
}

static inline void updateNumExamplesPerLabel(const CsrLabelMatrix& labelMatrix, uint32 exampleIndex,
                                             uint32* numExamplesPerLabel,
                                             std::unordered_map<uint32, uint32>& affectedLabelIndices) {
    CsrLabelMatrix::index_const_iterator indexIterator = labelMatrix.row_indices_cbegin(exampleIndex);
    uint32 numLabels = labelMatrix.row_indices_cend(exampleIndex) - indexIterator;

    for (uint32 i = 0; i < numLabels; i++) {
        uint32 labelIndex = indexIterator[i];
        uint32 numRemaining = numExamplesPerLabel[labelIndex];
        numExamplesPerLabel[labelIndex] = numRemaining - 1;
        affectedLabelIndices.emplace(labelIndex, numRemaining);
    }
}

/**
 * Implements iterative stratified sampling for selecting a subset of the available training examples, such that for
 * each label the proportion of relevant and irrelevant examples is maintained.
 *
 * @tparam LabelMatrix      The type of the label matrix that provides random or row-wise access to the labels of the
 *                          training examples
 * @tparam IndexIterator    The type of the iterator that provides access to the indices of the examples that are
 *                          contained by the training set
 */
template<class LabelMatrix, class IndexIterator>
class LabelWiseStratifiedSampling final : public IInstanceSubSampling {

    private:

        float32 sampleSize_;

        uint32 numTotalSamples_;

        uint32 numTotalOutOfSamples_;

        DenseWeightVector<uint8> weightVector_;

        uint32* rowIndices_;

        uint32* colIndices_;

        uint32 numCols_;

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
        LabelWiseStratifiedSampling(const LabelMatrix& labelMatrix, IndexIterator indicesBegin,
                                    IndexIterator indicesEnd, float32 sampleSize)
            : sampleSize_(sampleSize), numTotalSamples_((uint32) std::round(sampleSize * (indicesEnd - indicesBegin))),
              numTotalOutOfSamples_((indicesEnd - indicesBegin) - numTotalSamples_),
              weightVector_(DenseWeightVector<uint8>(labelMatrix.getNumRows(), true)) {
            // Convert the given label matrix into the CSC format...
            const CscLabelMatrix cscLabelMatrix(labelMatrix, indicesBegin, indicesEnd);

            // Create an array that stores for each label the number of examples that are associated with the label, as
            // well as a sorted map that stores all label indices in increasing order of the number of associated
            // examples...
            uint32 numLabels = cscLabelMatrix.getNumCols();
            uint32 numExamplesPerLabel[numLabels];
            std::multimap<uint32, uint32> sortedLabelIndices;

            for (uint32 i = 0; i < numLabels; i++) {
                uint32 numExamples = cscLabelMatrix.column_indices_cend(i) - cscLabelMatrix.column_indices_cbegin(i);
                numExamplesPerLabel[i] = numExamples;

                if (numExamples > 0) {
                    sortedLabelIndices.emplace(numExamples, i);
                }
            }

            // Allocate arrays for storing the row and column indices of the labels to be processed by the sampling
            // method in the CSC format...
            rowIndices_ = (uint32*) malloc(cscLabelMatrix.getNumNonZeroElements() * sizeof(uint32));
            colIndices_ = (uint32*) malloc((sortedLabelIndices.size() + 1) * sizeof(uint32));
            uint32 numNonZeroElements = 0;
            uint32 numCols = 0;

            // As long as there are labels that have not been processed yet, proceed with the label that has the
            // smallest number of associated examples...
            DenseWeightVector<uint8>::iterator weightIterator = weightVector_.begin();
            std::unordered_map<uint32, uint32> affectedLabelIndices;
            std::multimap<uint32, uint32>::iterator firstEntry;

            while ((firstEntry = sortedLabelIndices.begin()) != sortedLabelIndices.end()) {
                uint32 labelIndex = firstEntry->second;

                // Remove the label from the sorted map...
                sortedLabelIndices.erase(firstEntry);

                // Add the number of non-zero labels that have been processed so far to the array of column indices...
                colIndices_[numCols] = numNonZeroElements;
                numCols++;

                // Iterate the examples that are associated with the current label, if no weight has been set yet...
                CscLabelMatrix::index_const_iterator indexIterator = cscLabelMatrix.column_indices_cbegin(labelIndex);
                uint32 numExamples = cscLabelMatrix.column_indices_cend(labelIndex) - indexIterator;

                for (uint32 i = 0; i < numExamples; i++) {
                    uint32 exampleIndex = indexIterator[i];

                    // If the example's weight is 0, it has not been encountered yet...
                    if (weightIterator[exampleIndex] == 0) {
                        // Set the example's weight to 1...
                        weightIterator[exampleIndex] = 1;

                        // Add the example's index to the array of row indices...
                        rowIndices_[numNonZeroElements] = exampleIndex;
                        numNonZeroElements++;

                        // For each label that is associated with the example, decrement the number of associated
                        // examples by one...
                        updateNumExamplesPerLabel(labelMatrix, exampleIndex, &numExamplesPerLabel[0],
                                                  affectedLabelIndices);
                    }
                }

                // Remove each label, for which the number of associated examples have been changed previously, from the
                // sorted map and add it again to update the order...
                for (auto it = affectedLabelIndices.cbegin(); it != affectedLabelIndices.cend(); it++) {
                    uint32 key = it->first;

                    if (key != labelIndex) {
                        uint32 value = it->second;
                        auto range = sortedLabelIndices.equal_range(value);

                        for (auto it2 = range.first; it2 != range.second; it2++) {
                            if (it2->second == key) {
                                uint32 numRemaining = numExamplesPerLabel[key];

                                if (numRemaining > 0) {
                                    sortedLabelIndices.emplace_hint(it2, numRemaining, key);
                                }

                                sortedLabelIndices.erase(it2);
                                break;
                            }
                        }
                    }
                }

                affectedLabelIndices.clear();
            }

            // If there are examples that are not associated with any labels, we handle them separately..
            uint32 numTrainingExamples = weightVector_.getNumElements();
            uint32 numRemaining = numTrainingExamples - numNonZeroElements;

            if (numRemaining > 0) {
                // Adjust the size of the arrays that are used to store row and column indices...
                rowIndices_ = (uint32*) realloc(rowIndices_, (numNonZeroElements + numRemaining) * sizeof(uint32));
                colIndices_ = (uint32*) realloc(colIndices_, (numCols + 1) * sizeof(uint32));

                // Add the number of non-zero labels that have been processed so far to the array of column indices...
                colIndices_[numCols] = numNonZeroElements;
                numCols++;

                // Iterate the weights of all examples to find those whose weight has not been set yet...
                for (uint32 i = 0; i < numTrainingExamples; i++) {
                    if (weightIterator[i] == 0) {
                        // Add the example's index to the array of row indices...
                        rowIndices_[numNonZeroElements] = i;
                        numNonZeroElements++;
                    }
                }
            } else {
                // Adjust the size of the arrays that are used to store row and column indices...
                rowIndices_ = (uint32*) realloc(rowIndices_, numNonZeroElements * sizeof(uint32));
                colIndices_ = (uint32*) realloc(colIndices_, numCols * sizeof(uint32));
            }

            colIndices_[numCols - 1] = numNonZeroElements;
            numCols_ = numCols - 1;
        }

        const IWeightVector& subSample(RNG& rng) override {
            DenseWeightVector<uint8>::iterator weightIterator = weightVector_.begin();
            uint32 numNonZeroWeights = 0;
            uint32 numZeroWeights = 0;

            // For each column, assign a weight to the corresponding examples...
            for (uint32 i = 0; i < numCols_; i++) {
                uint32 start = colIndices_[i];
                uint32* exampleIndices = &rowIndices_[start];
                uint32 end = colIndices_[i + 1];
                uint32 numExamples = end - start;
                float32 numSamplesDecimal = sampleSize_ * numExamples;
                uint32 numDesiredSamples = numTotalSamples_ - numNonZeroWeights;
                uint32 numDesiredOutOfSamples = numTotalOutOfSamples_ - numZeroWeights;
                uint32 numSamples = (uint32) (tiebreak(numDesiredSamples, numDesiredOutOfSamples, rng) ?
                                              std::ceil(numSamplesDecimal) : std::floor(numSamplesDecimal));
                numNonZeroWeights += numSamples;
                numZeroWeights =+ (numExamples - numSamples);
                uint32 j;

                // Use the Fisher-Yates shuffle to randomly draw `numSamples` examples and set their weights to 1...
                for (j = 0; j < numExamples; j++) {
                    uint32 randomIndex = rng.random(j, numExamples);
                    uint32 exampleIndex = exampleIndices[randomIndex];
                    exampleIndices[randomIndex] = exampleIndices[j];
                    exampleIndices[j] = exampleIndex;
                    weightIterator[exampleIndex] = 1;
                    numSamples--;

                    if (numSamples == 0) {
                        break;
                    }
                }

                // Set the weights of the remaining examples to 0...
                for (j = j + 1; j < numExamples; j++) {
                    uint32 exampleIndex = exampleIndices[j];
                    weightIterator[exampleIndex] = 0;
                }
            }

            weightVector_.setNumNonZeroWeights(numNonZeroWeights);
            return weightVector_;
        }

};

LabelWiseStratifiedSamplingFactory::LabelWiseStratifiedSamplingFactory(float32 sampleSize)
    : sampleSize_(sampleSize) {

}

std::unique_ptr<IInstanceSubSampling> LabelWiseStratifiedSamplingFactory::create(
        const CContiguousLabelMatrix& labelMatrix, const SinglePartition& partition) const {
    return std::make_unique<LabelWiseStratifiedSampling<CContiguousLabelMatrix, SinglePartition::const_iterator>>(
        labelMatrix, partition.cbegin(), partition.cend(), sampleSize_);
}

std::unique_ptr<IInstanceSubSampling> LabelWiseStratifiedSamplingFactory::create(
        const CContiguousLabelMatrix& labelMatrix, BiPartition& partition) const {
    return std::make_unique<LabelWiseStratifiedSampling<CContiguousLabelMatrix, BiPartition::const_iterator>>(
        labelMatrix, partition.first_cbegin(), partition.first_cend(), sampleSize_);
}

std::unique_ptr<IInstanceSubSampling> LabelWiseStratifiedSamplingFactory::create(
        const CsrLabelMatrix& labelMatrix, const SinglePartition& partition) const {
    return std::make_unique<LabelWiseStratifiedSampling<CsrLabelMatrix, SinglePartition::const_iterator>>(
        labelMatrix, partition.cbegin(), partition.cend(), sampleSize_);
}

std::unique_ptr<IInstanceSubSampling> LabelWiseStratifiedSamplingFactory::create(
        const CsrLabelMatrix& labelMatrix, BiPartition& partition) const {
    return std::make_unique<LabelWiseStratifiedSampling<CsrLabelMatrix, BiPartition::const_iterator>>(
        labelMatrix, partition.first_cbegin(), partition.first_cend(), sampleSize_);
}
