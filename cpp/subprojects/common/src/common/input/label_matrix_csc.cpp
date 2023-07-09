#include "common/input/label_matrix_csc.hpp"

#include "common/data/arrays.hpp"

template<typename IndexIterator>
static inline uint32 copyLabelMatrix(uint32* rowIndices, uint32* indptr,
                                     const CContiguousConstView<const uint8>& labelMatrix, IndexIterator indicesBegin,
                                     IndexIterator indicesEnd) {
    uint32 numExamples = indicesEnd - indicesBegin;
    uint32 numLabels = labelMatrix.getNumCols();
    uint32 n = 0;

    for (uint32 i = 0; i < numLabels; i++) {
        indptr[i] = n;

        for (uint32 j = 0; j < numExamples; j++) {
            uint32 exampleIndex = indicesBegin[j];

            if (labelMatrix.values_cbegin(exampleIndex)[i]) {
                rowIndices[n] = exampleIndex;
                n++;
            }
        }
    }

    return n;
}

template<typename IndexIterator>
static inline uint32 copyLabelMatrix(uint32* rowIndices, uint32* indptr, const BinaryCsrConstView& labelMatrix,
                                     IndexIterator indicesBegin, IndexIterator indicesEnd) {
    uint32 numExamples = indicesEnd - indicesBegin;
    uint32 numLabels = labelMatrix.getNumCols();

    // Set column indices of the CSC matrix to zero...
    setArrayToZeros(indptr, numLabels);

    // Determine the number of non-zero elements per column...
    for (uint32 i = 0; i < numExamples; i++) {
        uint32 exampleIndex = indicesBegin[i];
        BinaryCsrConstView::index_const_iterator labelIndexIterator = labelMatrix.indices_cbegin(exampleIndex);
        uint32 numRelevantLabels = labelMatrix.indices_cend(exampleIndex) - labelIndexIterator;

        for (uint32 j = 0; j < numRelevantLabels; j++) {
            uint32 labelIndex = labelIndexIterator[j];
            indptr[labelIndex]++;
        }
    }

    // Update the column indices of the CSC matrix with respect to the number of non-zero elements that correspond to
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
        BinaryCsrConstView::index_const_iterator labelIndexIterator = labelMatrix.indices_cbegin(exampleIndex);
        uint32 numRelevantLabels = labelMatrix.indices_cend(exampleIndex) - labelIndexIterator;

        for (uint32 j = 0; j < numRelevantLabels; j++) {
            uint32 originalLabelIndex = labelIndexIterator[j];
            uint32 labelIndex = indptr[originalLabelIndex];
            rowIndices[labelIndex] = exampleIndex;
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

    return tmp;
}

CscLabelMatrix::CscLabelMatrix(const CContiguousConstView<const uint8>& labelMatrix,
                               CompleteIndexVector::const_iterator indicesBegin,
                               CompleteIndexVector::const_iterator indicesEnd)
    : BinaryCscMatrix((uint32) (indicesEnd - indicesBegin), labelMatrix.getNumCols(),
                      ((uint32) (indicesEnd - indicesBegin)) * labelMatrix.getNumCols()) {
    uint32 numNonZeroElements = copyLabelMatrix<CompleteIndexVector::const_iterator>(
      this->rowIndices_, this->indptr_, labelMatrix, indicesBegin, indicesEnd);
    this->setNumNonZeroElements(numNonZeroElements, true);
}

CscLabelMatrix::CscLabelMatrix(const CContiguousConstView<const uint8>& labelMatrix,
                               PartialIndexVector::const_iterator indicesBegin,
                               PartialIndexVector::const_iterator indicesEnd)
    : BinaryCscMatrix((uint32) (indicesEnd - indicesBegin), labelMatrix.getNumCols(),
                      ((uint32) (indicesEnd - indicesBegin)) * labelMatrix.getNumCols()) {
    uint32 numNonZeroElements = copyLabelMatrix<PartialIndexVector::const_iterator>(
      this->rowIndices_, this->indptr_, labelMatrix, indicesBegin, indicesEnd);
    this->setNumNonZeroElements(numNonZeroElements, true);
}

CscLabelMatrix::CscLabelMatrix(const BinaryCsrConstView& labelMatrix, CompleteIndexVector::const_iterator indicesBegin,
                               CompleteIndexVector::const_iterator indicesEnd)
    : BinaryCscMatrix((uint32) (indicesEnd - indicesBegin), labelMatrix.getNumCols(),
                      labelMatrix.getNumNonZeroElements()) {
    uint32 numNonZeroElements = copyLabelMatrix<CompleteIndexVector::const_iterator>(
      this->rowIndices_, this->indptr_, labelMatrix, indicesBegin, indicesEnd);
    this->setNumNonZeroElements(numNonZeroElements, true);
}

CscLabelMatrix::CscLabelMatrix(const BinaryCsrConstView& labelMatrix, PartialIndexVector::const_iterator indicesBegin,
                               PartialIndexVector::const_iterator indicesEnd)
    : BinaryCscMatrix((uint32) (indicesEnd - indicesBegin), labelMatrix.getNumCols(),
                      labelMatrix.getNumNonZeroElements()) {
    uint32 numNonZeroElements = copyLabelMatrix<PartialIndexVector::const_iterator>(
      this->rowIndices_, this->indptr_, labelMatrix, indicesBegin, indicesEnd);
    this->setNumNonZeroElements(numNonZeroElements, true);
}
