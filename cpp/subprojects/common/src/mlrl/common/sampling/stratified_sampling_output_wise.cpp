#include "mlrl/common/sampling/stratified_sampling_output_wise.hpp"

#include "mlrl/common/data/array.hpp"
#include "mlrl/common/data/indexed_value.hpp"
#include "mlrl/common/sampling/partition_single.hpp"
#include "stratified_sampling_common.hpp"

#include <set>
#include <unordered_map>

template<typename IndexIterator>
static inline uint32* copyLabelMatrix(uint32* indices, uint32* indptr, const CContiguousView<const uint8>& labelMatrix,
                                      IndexIterator indicesBegin, IndexIterator indicesEnd) {
    uint32 numExamples = indicesEnd - indicesBegin;
    uint32 numLabels = labelMatrix.numCols;
    uint32 n = 0;

    for (uint32 i = 0; i < numLabels; i++) {
        indptr[i] = n;

        for (uint32 j = 0; j < numExamples; j++) {
            uint32 exampleIndex = indicesBegin[j];

            if (labelMatrix.values_cbegin(exampleIndex)[i]) {
                indices[n] = exampleIndex;
                n++;
            }
        }
    }

    indptr[numLabels] = n;
    return util::reallocateMemory(indices, n);
}

template<typename IndexIterator>
static inline uint32* copyLabelMatrix(uint32* indices, uint32* indptr, const BinaryCsrView& labelMatrix,
                                      IndexIterator indicesBegin, IndexIterator indicesEnd) {
    uint32 numExamples = indicesEnd - indicesBegin;
    uint32 numLabels = labelMatrix.numCols;

    // Set column indices of the CSC matrix to zero...
    util::setViewToZeros(indptr, numLabels);

    // Determine the number of dense elements per column...
    for (uint32 i = 0; i < numExamples; i++) {
        uint32 exampleIndex = indicesBegin[i];
        BinaryCsrView::index_const_iterator labelIndexIterator = labelMatrix.indices_cbegin(exampleIndex);
        uint32 numRelevantLabels = labelMatrix.indices_cend(exampleIndex) - labelIndexIterator;

        for (uint32 j = 0; j < numRelevantLabels; j++) {
            uint32 labelIndex = labelIndexIterator[j];
            indptr[labelIndex]++;
        }
    }

    // Update the column indices of the CSC matrix with respect to the number of dense elements that correspond to
    // previous columns...
    uint32 tmp = 0;

    for (uint32 i = 0; i < numLabels; i++) {
        uint32 labelIndex = indptr[i];
        indptr[i] = tmp;
        tmp += labelIndex;
    }

    // Set the row indices of the CSC matrix. This will modify the column indices...
    for (uint32 i = 0; i < numExamples; i++) {
        uint32 exampleIndex = indicesBegin[i];
        BinaryCsrView::index_const_iterator labelIndexIterator = labelMatrix.indices_cbegin(exampleIndex);
        uint32 numRelevantLabels = labelMatrix.indices_cend(exampleIndex) - labelIndexIterator;

        for (uint32 j = 0; j < numRelevantLabels; j++) {
            uint32 originalLabelIndex = labelIndexIterator[j];
            uint32 labelIndex = indptr[originalLabelIndex];
            indices[labelIndex] = exampleIndex;
            indptr[originalLabelIndex]++;
        }
    }

    // Reset the column indices to the previous values...
    tmp = 0;

    for (uint32 i = 0; i < numLabels; i++) {
        uint32 labelIndex = indptr[i];
        indptr[i] = tmp;
        tmp = labelIndex;
    }

    indptr[numLabels] = tmp;
    return util::reallocateMemory(indices, tmp);
}

/**
 * A two-dimensional view that provides column-wise read and write access to the labels of individual training examples
 * stored in a matrix in the compressed sparse column (CSC) format that have been copied from a `CContiguousView` or a
 * `BinaryCsrView`.
 */
class CscLabelMatrix final : public AllocatedBinaryCscView {
    public:

        /**
         * @param labelMatrix   A reference to an object of type `CContiguousView` to be copied
         * @param indicesBegin  A `CompleteIndexVector::const_iterator` to the beginning of the indices of the examples
         *                      to be considered
         * @param indicesEnd    A `CompleteIndexVector::const_iterator` to the end of the indices of the examples to be
         *                      considered
         */
        CscLabelMatrix(const CContiguousView<const uint8>& labelMatrix,
                       CompleteIndexVector::const_iterator indicesBegin, CompleteIndexVector::const_iterator indicesEnd)
            : AllocatedBinaryCscView((indicesEnd - indicesBegin) * labelMatrix.numCols, labelMatrix.numRows,
                                     labelMatrix.numCols) {
            BinarySparseMatrix::indices = copyLabelMatrix(BinarySparseMatrix::indices, BinarySparseMatrix::indptr,
                                                          labelMatrix, indicesBegin, indicesEnd);
        }

        /**
         * @param labelMatrix   A reference to an object of type `CContiguousView` to be copied
         * @param indicesBegin  A `PartialIndexVector::const_iterator` to the beginning of the indices of the examples
         *                      to be considered
         * @param indicesEnd    A `PartialIndexVector::const_iterator` to the end of the indices of the examples to be
         *                      considered
         */
        CscLabelMatrix(const CContiguousView<const uint8>& labelMatrix, PartialIndexVector::const_iterator indicesBegin,
                       PartialIndexVector::const_iterator indicesEnd)
            : AllocatedBinaryCscView((indicesEnd - indicesBegin) * labelMatrix.numCols, labelMatrix.numRows,
                                     labelMatrix.numCols) {
            BinarySparseMatrix::indices = copyLabelMatrix(BinarySparseMatrix::indices, BinarySparseMatrix::indptr,
                                                          labelMatrix, indicesBegin, indicesEnd);
        }

        /**
         * @param labelMatrix   A reference to an object of type `BinaryCsrView` to be copied
         * @param indicesBegin  A `CompleteIndexVector::const_iterator` to the beginning of the indices of the examples
         *                      to be considered
         * @param indicesEnd    A `CompleteIndexVector::const_iterator` to the end of the indices of the examples to be
         *                      considered
         */
        CscLabelMatrix(const BinaryCsrView& labelMatrix, CompleteIndexVector::const_iterator indicesBegin,
                       CompleteIndexVector::const_iterator indicesEnd)
            : AllocatedBinaryCscView(labelMatrix.getNumDenseElements(), labelMatrix.numRows, labelMatrix.numCols) {
            BinarySparseMatrix::indices = copyLabelMatrix(BinarySparseMatrix::indices, BinarySparseMatrix::indptr,
                                                          labelMatrix, indicesBegin, indicesEnd);
        }

        /**
         * @param labelMatrix   A reference to an object of type `BinaryCsrView` to be copied
         * @param indicesBegin  A `PartialIndexVector::const_iterator` to the beginning of the indices of the examples
         *                      to be considered
         * @param indicesEnd    A `PartialIndexVector::const_iterator` to the end of the indices of the examples to be
         *                      considered
         */
        CscLabelMatrix(const BinaryCsrView& labelMatrix, PartialIndexVector::const_iterator indicesBegin,
                       PartialIndexVector::const_iterator indicesEnd)
            : AllocatedBinaryCscView(labelMatrix.getNumDenseElements(), labelMatrix.numRows, labelMatrix.numCols) {
            BinarySparseMatrix::indices = copyLabelMatrix(BinarySparseMatrix::indices, BinarySparseMatrix::indptr,
                                                          labelMatrix, indicesBegin, indicesEnd);
        }

        /**
         * @param other A reference to an object of type `CscLabelMatrix` that should be moved
         */
        CscLabelMatrix(CscLabelMatrix&& other) : AllocatedBinaryCscView(std::move(other)) {}
};

/**
 * Allows to compare two objects of type `IndexedValue` according to the following strict weak ordering: If the value of
 * the first object is smaller, it goes before the second one. If the values of both objects are equal and the index of
 * the first object is smaller, it goes before the second one. Otherwise, the first object goes after the second one.
 */
struct CompareIndexedValue final {
    public:

        /**
         * Returns whether the a given object of type `IndexedValue` should go before a second one.
         *
         * @param lhs   A reference to a first object of type `IndexedValue`
         * @param rhs   A reference to a second object of type `IndexedValue`
         * @return      True, if the first object should go before the second one, false otherwise
         */
        inline bool operator()(const IndexedValue<uint32>& lhs, const IndexedValue<uint32>& rhs) const {
            return lhs.value < rhs.value || (lhs.value == rhs.value && lhs.index < rhs.index);
        }
};

static inline void updateNumExamplesPerLabel(const CContiguousView<const uint8>& labelMatrix, uint32 exampleIndex,
                                             Array<uint32>& numExamplesPerLabel,
                                             std::unordered_map<uint32, uint32>& affectedLabelIndices) {
    CContiguousView<const uint8>::value_const_iterator labelIterator = labelMatrix.values_cbegin(exampleIndex);
    uint32 numLabels = labelMatrix.numCols;

    for (uint32 i = 0; i < numLabels; i++) {
        if (labelIterator[i]) {
            uint32 numRemaining = numExamplesPerLabel[i];
            numExamplesPerLabel[i] = numRemaining - 1;
            affectedLabelIndices.emplace(i, numRemaining);
        }
    }
}

static inline void updateNumExamplesPerLabel(const BinaryCsrView& labelMatrix, uint32 exampleIndex,
                                             Array<uint32>& numExamplesPerLabel,
                                             std::unordered_map<uint32, uint32>& affectedLabelIndices) {
    BinaryCsrView::index_const_iterator indexIterator = labelMatrix.indices_cbegin(exampleIndex);
    uint32 numLabels = labelMatrix.indices_cend(exampleIndex) - indexIterator;

    for (uint32 i = 0; i < numLabels; i++) {
        uint32 labelIndex = indexIterator[i];
        uint32 numRemaining = numExamplesPerLabel[labelIndex];
        numExamplesPerLabel[labelIndex] = numRemaining - 1;
        affectedLabelIndices.emplace(labelIndex, numRemaining);
    }
}

/**
 * A two-dimensional view that provides column-wise read and write access to the indices of individual training examples
 * stored in a matrix in the compressed sparse column (CSC) format. The indices in each column of the matrix correspond
 * to the examples to be sampled for a particular label.
 *
 * @tparam LabelMatrix      The type of the label matrix that provides random or row-wise access to the labels of the
 *                          training examples
 * @tparam IndexIterator    The type of the iterator that provides access to the indices of the examples that should be
 *                          considered
 */
template<typename LabelMatrix, typename IndexIterator>
class StratificationMatrix final : public AllocatedBinaryCscView {
    public:

        /**
         * @param rowWiseLabelMatrix    A reference to an on object of template type `LabelMatrix` that provides random
         *                              or row-wise access to the labels of the training examples
         * @param columnWiseLabelMatrix A reference to an object of type `CscLabelMatrix` that provides column-wise
         *                              access to the labels of the training examples
         * @param indicesBegin          An iterator to the beginning of the indices of the examples that should be
         *                              considered
         * @param indicesEnd            An iterator to the end of the indices of the examples that should be considered
         */
        StratificationMatrix(const LabelMatrix& rowWiseLabelMatrix, const CscLabelMatrix& columnWiseLabelMatrix,
                             IndexIterator indicesBegin, IndexIterator indicesEnd)
            : AllocatedBinaryCscView(columnWiseLabelMatrix.getNumDenseElements(), indicesEnd - indicesBegin,
                                     columnWiseLabelMatrix.numCols) {
            // Create an array that stores for each label the number of examples that are associated with the label, as
            // well as a sorted map that stores all label indices in increasing order of the number of associated
            // examples...
            Array<uint32> numExamplesPerLabel(Matrix::numCols);
            typedef std::set<IndexedValue<uint32>, CompareIndexedValue> SortedSet;
            SortedSet sortedLabelIndices;

            for (uint32 i = 0; i < Matrix::numCols; i++) {
                uint32 numExamples = columnWiseLabelMatrix.indices_cend(i) - columnWiseLabelMatrix.indices_cbegin(i);
                numExamplesPerLabel[i] = numExamples;

                if (numExamples > 0) {
                    sortedLabelIndices.emplace(i, numExamples);
                }
            }

            // Allocate arrays for storing the row and column indices of the labels to be processed by the sampling
            // method in the CSC format...
            uint32 numDenseElements = 0;
            uint32 numCols = 0;

            // Create a boolean array that stores whether individual examples remain to be processed (1) or not (0)...
            uint32 numTotalExamples = columnWiseLabelMatrix.numRows;
            BitVector mask(numTotalExamples, true);

            for (uint32 i = 0; i < Matrix::numRows; i++) {
                uint32 exampleIndex = indicesBegin[i];
                mask.set(exampleIndex, true);
            }

            // As long as there are labels that have not been processed yet, proceed with the label that has the
            // smallest number of associated examples...
            std::unordered_map<uint32, uint32> affectedLabelIndices;
            SortedSet::iterator firstEntry;

            while ((firstEntry = sortedLabelIndices.begin()) != sortedLabelIndices.end()) {
                const IndexedValue<uint32>& entry = *firstEntry;
                uint32 labelIndex = entry.index;

                // Remove the label from the sorted map...
                sortedLabelIndices.erase(firstEntry);

                // Add the number of dense elements that have been processed so far to the array of column indices...
                BinarySparseMatrix::indptr[numCols] = numDenseElements;
                numCols++;

                // Iterate the examples that are associated with the current label, if no weight has been set yet...
                CscLabelMatrix::index_const_iterator indexIterator = columnWiseLabelMatrix.indices_cbegin(labelIndex);
                uint32 numExamples = columnWiseLabelMatrix.indices_cend(labelIndex) - indexIterator;

                for (uint32 i = 0; i < numExamples; i++) {
                    uint32 exampleIndex = indexIterator[i];

                    // If the example has not been processed yet...
                    if (mask[exampleIndex]) {
                        mask.set(exampleIndex, false);

                        // Add the example's index to the array of row indices...
                        BinarySparseMatrix::indices[numDenseElements] = exampleIndex;
                        numDenseElements++;

                        // For each label that is associated with the example, decrement the number of associated
                        // examples by one...
                        updateNumExamplesPerLabel(rowWiseLabelMatrix, exampleIndex, numExamplesPerLabel,
                                                  affectedLabelIndices);
                    }
                }

                // Remove each label, for which the number of associated examples have been changed previously, from the
                // sorted map and add it again to update the order...
                for (auto it = affectedLabelIndices.cbegin(); it != affectedLabelIndices.cend(); it++) {
                    uint32 key = it->first;

                    if (key != labelIndex) {
                        uint32 value = it->second;
                        SortedSet::iterator it2 = sortedLabelIndices.find(IndexedValue<uint32>(key, value));
                        uint32 numRemaining = numExamplesPerLabel[key];

                        if (numRemaining > 0) {
                            sortedLabelIndices.emplace_hint(it2, key, numRemaining);
                        }

                        sortedLabelIndices.erase(it2);
                    }
                }

                affectedLabelIndices.clear();
            }

            // If there are examples that are not associated with any labels, we handle them separately..
            uint32 numRemaining = Matrix::numRows - numDenseElements;

            if (numRemaining > 0) {
                // Adjust the size of the arrays that are used to store row and column indices...
                BinarySparseMatrix::indices =
                  util::reallocateMemory(BinarySparseMatrix::indices, numDenseElements + numRemaining);
                BinarySparseMatrix::indptr = util::reallocateMemory(BinarySparseMatrix::indptr, numCols + 2);

                // Add the number of dense elements that have been processed so far to the array of column indices...
                BinarySparseMatrix::indptr[numCols] = numDenseElements;
                numCols++;

                // Iterate the weights of all examples to find those whose weight has not been set yet...
                for (uint32 i = 0; i < numTotalExamples; i++) {
                    if (mask[i]) {
                        // Add the example's index to the array of row indices...
                        BinarySparseMatrix::indices[numDenseElements] = i;
                        numDenseElements++;
                    }
                }
            } else {
                // Adjust the size of the arrays that are used to store row and column indices...
                BinarySparseMatrix::indices = util::reallocateMemory(BinarySparseMatrix::indices, numDenseElements);
                BinarySparseMatrix::indptr = util::reallocateMemory(BinarySparseMatrix::indptr, numCols + 1);
            }

            BinarySparseMatrix::indptr[numCols] = numDenseElements;
            Matrix::numCols = numCols;
        }

        /**
         * @param other A reference to an object of type `StratificationMatrix` that should be moved
         */
        StratificationMatrix(StratificationMatrix<LabelMatrix, IndexIterator>&& other)
            : AllocatedBinaryCscView(std::move(other)) {}
};

template<typename LabelMatrix, typename IndexIterator>
LabelWiseStratification<LabelMatrix, IndexIterator>::LabelWiseStratification(const LabelMatrix& labelMatrix,
                                                                             IndexIterator indicesBegin,
                                                                             IndexIterator indicesEnd)
    : stratificationMatrix_(StratificationMatrix<LabelMatrix, IndexIterator>(
        labelMatrix, CscLabelMatrix(labelMatrix, indicesBegin, indicesEnd), indicesBegin, indicesEnd)) {}

template<typename LabelMatrix, typename IndexIterator>
void LabelWiseStratification<LabelMatrix, IndexIterator>::sampleWeights(BitWeightVector& weightVector,
                                                                        float32 sampleSize, RNG& rng) {
    uint32 numRows = stratificationMatrix_.getNumRows();
    uint32 numCols = stratificationMatrix_.getNumCols();
    uint32 numTotalSamples = static_cast<uint32>(std::round(sampleSize * numRows));
    uint32 numTotalOutOfSamples = numRows - numTotalSamples;
    uint32 numNonZeroWeights = 0;
    uint32 numZeroWeights = 0;

    // For each column, assign a weight to the corresponding examples...
    for (uint32 i = 0; i < numCols; i++) {
        auto exampleIndices = stratificationMatrix_.indices_begin(i);
        uint32 numExamples = stratificationMatrix_.indices_end(i) - exampleIndices;
        float32 numSamplesDecimal = sampleSize * numExamples;
        uint32 numDesiredSamples = numTotalSamples - numNonZeroWeights;
        uint32 numDesiredOutOfSamples = numTotalOutOfSamples - numZeroWeights;
        uint32 numSamples =
          static_cast<uint32>(tiebreak(numDesiredSamples, numDesiredOutOfSamples, rng) ? std::ceil(numSamplesDecimal)
                                                                                       : std::floor(numSamplesDecimal));
        numNonZeroWeights += numSamples;
        numZeroWeights += (numExamples - numSamples);
        uint32 j;

        // Use the Fisher-Yates shuffle to randomly draw `numSamples` examples and set their weights to 1...
        for (j = 0; j < numSamples; j++) {
            uint32 randomIndex = rng.random(j, numExamples);
            uint32 exampleIndex = exampleIndices[randomIndex];
            exampleIndices[randomIndex] = exampleIndices[j];
            exampleIndices[j] = exampleIndex;
            weightVector.set(exampleIndex, true);
        }

        // Set the weights of the remaining examples to 0...
        for (; j < numExamples; j++) {
            uint32 exampleIndex = exampleIndices[j];
            weightVector.set(exampleIndex, false);
        }
    }

    weightVector.setNumNonZeroWeights(numNonZeroWeights);
}

template<typename LabelMatrix, typename IndexIterator>
void LabelWiseStratification<LabelMatrix, IndexIterator>::sampleBiPartition(BiPartition& partition, RNG& rng) {
    uint32 numCols = stratificationMatrix_.getNumCols();
    BiPartition::iterator firstIterator = partition.first_begin();
    BiPartition::iterator secondIterator = partition.second_begin();
    uint32 numFirst = partition.getNumFirst();
    uint32 numSecond = partition.getNumSecond();

    for (uint32 i = 0; i < numCols; i++) {
        auto exampleIndices = stratificationMatrix_.indices_begin(i);
        uint32 numExamples = stratificationMatrix_.indices_end(i) - exampleIndices;

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
        uint32 j;

        for (j = 0; j < numSamples; j++) {
            uint32 randomIndex = rng.random(j, numExamples);
            uint32 exampleIndex = exampleIndices[randomIndex];
            exampleIndices[randomIndex] = exampleIndices[j];
            exampleIndices[j] = exampleIndex;
            *firstIterator = exampleIndex;
            firstIterator++;
        }

        // Add the remaining examples to the second set...
        for (; j < numExamples; j++) {
            uint32 exampleIndex = exampleIndices[j];
            *secondIterator = exampleIndex;
            secondIterator++;
        }
    }
}

template class LabelWiseStratification<CContiguousView<const uint8>, SinglePartition::const_iterator>;
template class LabelWiseStratification<CContiguousView<const uint8>, BiPartition::const_iterator>;
template class LabelWiseStratification<BinaryCsrView, SinglePartition::const_iterator>;
template class LabelWiseStratification<BinaryCsrView, BiPartition::const_iterator>;
