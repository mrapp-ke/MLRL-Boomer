#include "mlrl/common/input/label_matrix_csc.hpp"

template<typename IndexIterator>
static inline uint32* copyLabelMatrix(uint32* rowIndices, uint32* indptr,
                                      const CContiguousView<const uint8>& labelMatrix, IndexIterator indicesBegin,
                                      IndexIterator indicesEnd) {
    uint32 numExamples = indicesEnd - indicesBegin;
    uint32 numLabels = labelMatrix.numCols;
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

    indptr[numLabels] = n;
    return reallocateMemory(rowIndices, n);
}

template<typename IndexIterator>
static inline uint32* copyLabelMatrix(uint32* rowIndices, uint32* indptr, const BinaryCsrView& labelMatrix,
                                      IndexIterator indicesBegin, IndexIterator indicesEnd) {
    uint32 numExamples = indicesEnd - indicesBegin;
    uint32 numLabels = labelMatrix.numCols;

    // Set column indices of the CSC matrix to zero...
    setViewToZeros(indptr, numLabels);

    // Determine the number of non-zero elements per column...
    for (uint32 i = 0; i < numExamples; i++) {
        uint32 exampleIndex = indicesBegin[i];
        BinaryCsrView::index_const_iterator labelIndexIterator = labelMatrix.indices_cbegin(exampleIndex);
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
        BinaryCsrView::index_const_iterator labelIndexIterator = labelMatrix.indices_cbegin(exampleIndex);
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

    indptr[numLabels] = tmp;
    return reallocateMemory(rowIndices, tmp);
}

CscLabelMatrix::CscLabelMatrix(const CContiguousView<const uint8>& labelMatrix,
                               CompleteIndexVector::const_iterator indicesBegin,
                               CompleteIndexVector::const_iterator indicesEnd)
    : BinaryCscView(allocateMemory<uint32>((indicesEnd - indicesBegin) * labelMatrix.numCols),
                    allocateMemory<uint32>(labelMatrix.numCols + 1), indicesEnd - indicesBegin, labelMatrix.numCols) {
    this->indices = copyLabelMatrix<CompleteIndexVector::const_iterator>(this->indices, this->indptr, labelMatrix,
                                                                         indicesBegin, indicesEnd);
}

CscLabelMatrix::CscLabelMatrix(const CContiguousView<const uint8>& labelMatrix,
                               PartialIndexVector::const_iterator indicesBegin,
                               PartialIndexVector::const_iterator indicesEnd)
    : BinaryCscView(allocateMemory<uint32>((indicesEnd - indicesBegin) * labelMatrix.numCols),
                    allocateMemory<uint32>(labelMatrix.numCols + 1), indicesEnd - indicesBegin, labelMatrix.numCols) {
    this->indices = copyLabelMatrix<PartialIndexVector::const_iterator>(this->indices, this->indptr, labelMatrix,
                                                                        indicesBegin, indicesEnd);
}

CscLabelMatrix::CscLabelMatrix(const BinaryCsrView& labelMatrix, CompleteIndexVector::const_iterator indicesBegin,
                               CompleteIndexVector::const_iterator indicesEnd)
    : BinaryCscView(allocateMemory<uint32>(labelMatrix.getNumNonZeroElements()),
                    allocateMemory<uint32>(labelMatrix.numCols + 1), indicesEnd - indicesBegin, labelMatrix.numCols) {
    this->indices = copyLabelMatrix<CompleteIndexVector::const_iterator>(this->indices, this->indptr, labelMatrix,
                                                                         indicesBegin, indicesEnd);
}

CscLabelMatrix::CscLabelMatrix(const BinaryCsrView& labelMatrix, PartialIndexVector::const_iterator indicesBegin,
                               PartialIndexVector::const_iterator indicesEnd)
    : BinaryCscView(allocateMemory<uint32>(labelMatrix.getNumNonZeroElements()),
                    allocateMemory<uint32>(labelMatrix.numCols + 1), indicesEnd - indicesBegin, labelMatrix.numCols) {
    this->indices = copyLabelMatrix<PartialIndexVector::const_iterator>(this->indices, this->indptr, labelMatrix,
                                                                        indicesBegin, indicesEnd);
}

CscLabelMatrix::~CscLabelMatrix() {
    freeMemory(this->indices);
    freeMemory(this->indptr);
}
