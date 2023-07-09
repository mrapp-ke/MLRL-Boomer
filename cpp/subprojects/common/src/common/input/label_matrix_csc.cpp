#include "common/input/label_matrix_csc.hpp"

#include "common/data/arrays.hpp"

template<typename IndexIterator>
static inline void copyLabelMatrix(BinaryCscMatrix& cscMatrix, const CContiguousConstView<const uint8>& labelMatrix,
                                   IndexIterator indexIterator) {
    uint32 numExamples = cscMatrix.getNumRows();
    uint32 numLabels = cscMatrix.getNumCols();
    BinaryCscMatrix::index_iterator indptrIterator = cscMatrix.indptr_begin();
    indptrIterator[0] = 0;
    BinaryCscMatrix::index_iterator rowIndexIterator = cscMatrix.indices_begin(0);
    uint32 n = 0;

    for (uint32 i = 0; i < numLabels; i++) {
        indptrIterator[i] = n;

        for (uint32 j = 0; j < numExamples; j++) {
            uint32 exampleIndex = indexIterator[j];

            if (labelMatrix.values_cbegin(exampleIndex)[i]) {
                rowIndexIterator[n] = exampleIndex;
                n++;
            }
        }
    }

    cscMatrix.setNumNonZeroElements(n, true);
}

template<typename IndexIterator>
static inline void copyLabelMatrix(BinaryCscMatrix& cscMatrix, const BinaryCsrConstView& labelMatrix,
                                   IndexIterator indexIterator) {
    uint32 numExamples = cscMatrix.getNumRows();
    uint32 numLabels = cscMatrix.getNumCols();
    BinaryCscMatrix::index_iterator indptrIterator = cscMatrix.indptr_begin();

    // Set column indices of the CSC matrix to zero...
    setArrayToZeros(indptrIterator, numLabels);

    // Determine the number of non-zero elements per column...
    for (uint32 i = 0; i < numExamples; i++) {
        uint32 exampleIndex = indexIterator[i];
        BinaryCsrConstView::index_const_iterator labelIndexIterator = labelMatrix.indices_cbegin(exampleIndex);
        uint32 numRelevantLabels = labelMatrix.indices_cend(exampleIndex) - labelIndexIterator;

        for (uint32 j = 0; j < numRelevantLabels; j++) {
            uint32 labelIndex = labelIndexIterator[j];
            indptrIterator[labelIndex]++;
        }
    }

    // Update the column indices of the CSC matrix with respect to the number of non-zero elements that correspond to
    // previous columns...
    uint32 tmp = 0;

    for (uint32 i = 0; i < numLabels; i++) {
        uint32 labelIndex = indptrIterator[i];
        indptrIterator[i] = tmp;
        tmp += labelIndex;
    }

    // Set the row indices of the CSC matrix. This will modify the column indices...
    BinaryCscMatrix::index_iterator rowIndexIterator = cscMatrix.indices_begin(0);

    for (uint32 i = 0; i < numExamples; i++) {
        uint32 exampleIndex = indexIterator[i];
        BinaryCsrConstView::index_const_iterator labelIndexIterator = labelMatrix.indices_cbegin(exampleIndex);
        uint32 numRelevantLabels = labelMatrix.indices_cend(exampleIndex) - labelIndexIterator;

        for (uint32 j = 0; j < numRelevantLabels; j++) {
            uint32 originalLabelIndex = labelIndexIterator[j];
            uint32 labelIndex = indptrIterator[originalLabelIndex];
            rowIndexIterator[labelIndex] = exampleIndex;
            indptrIterator[originalLabelIndex]++;
        }
    }

    // Reset the column indices to the previous values...
    tmp = 0;

    for (uint32 i = 0; i < numLabels; i++) {
        uint32 labelIndex = indptrIterator[i];
        indptrIterator[i] = tmp;
        tmp = labelIndex;
    }

    cscMatrix.setNumNonZeroElements(tmp, true);
}

CscLabelMatrix::CscLabelMatrix(const CContiguousConstView<const uint8>& labelMatrix,
                               CompleteIndexVector::const_iterator indicesBegin,
                               CompleteIndexVector::const_iterator indicesEnd)
    : BinaryCscMatrix((uint32) (indicesEnd - indicesBegin), labelMatrix.getNumCols(),
                      ((uint32) (indicesEnd - indicesBegin)) * labelMatrix.getNumCols()) {
    copyLabelMatrix<CompleteIndexVector::const_iterator>(*this, labelMatrix, indicesBegin);
}

CscLabelMatrix::CscLabelMatrix(const CContiguousConstView<const uint8>& labelMatrix,
                               PartialIndexVector::const_iterator indicesBegin,
                               PartialIndexVector::const_iterator indicesEnd)
    : BinaryCscMatrix((uint32) (indicesEnd - indicesBegin), labelMatrix.getNumCols(),
                      ((uint32) (indicesEnd - indicesBegin)) * labelMatrix.getNumCols()) {
    copyLabelMatrix<PartialIndexVector::const_iterator>(*this, labelMatrix, indicesBegin);
}

CscLabelMatrix::CscLabelMatrix(const BinaryCsrConstView& labelMatrix, CompleteIndexVector::const_iterator indicesBegin,
                               CompleteIndexVector::const_iterator indicesEnd)
    : BinaryCscMatrix((uint32) (indicesEnd - indicesBegin), labelMatrix.getNumCols(),
                      labelMatrix.getNumNonZeroElements()) {
    copyLabelMatrix<CompleteIndexVector::const_iterator>(*this, labelMatrix, indicesBegin);
}

CscLabelMatrix::CscLabelMatrix(const BinaryCsrConstView& labelMatrix, PartialIndexVector::const_iterator indicesBegin,
                               PartialIndexVector::const_iterator indicesEnd)
    : BinaryCscMatrix((uint32) (indicesEnd - indicesBegin), labelMatrix.getNumCols(),
                      labelMatrix.getNumNonZeroElements()) {
    copyLabelMatrix<PartialIndexVector::const_iterator>(*this, labelMatrix, indicesBegin);
}
