#include "common/input/label_matrix_csc.hpp"
#include "common/data/arrays.hpp"
#include <cstdlib>


CscLabelMatrix::CscLabelMatrix(const CContiguousLabelMatrix& labelMatrix)
    : rowIndices_((uint32*) malloc(labelMatrix.getNumRows() * labelMatrix.getNumCols() * sizeof(uint32))),
      colIndices_((uint32*) malloc((labelMatrix.getNumCols() + 1) * sizeof(uint32))),
      view_(BinaryCscView(labelMatrix.getNumRows(), labelMatrix.getNumCols(), rowIndices_, colIndices_)) {
    uint32 numRows = this->getNumRows();
    uint32 numCols = this->getNumCols();
    uint32 n = 0;

    for (uint32 i = 0; i < numCols; i++) {
        colIndices_[i] = n;

        for (uint32 j = 0; j < numRows; j++) {
            if (labelMatrix.row_values_cbegin(j)[i]) {
                rowIndices_[n] = j;
                n++;
            }
        }
    }

    colIndices_[numCols] = n;
    rowIndices_ = (uint32*) realloc(rowIndices_, n * sizeof(uint32));
}

CscLabelMatrix::CscLabelMatrix(const CsrLabelMatrix& labelMatrix)
    : rowIndices_((uint32*) malloc(labelMatrix.getNumNonZeroElements() * sizeof(uint32))),
      colIndices_((uint32*) malloc((labelMatrix.getNumCols() + 1) * sizeof(uint32))),
      view_(BinaryCscView(labelMatrix.getNumRows(), labelMatrix.getNumCols(), rowIndices_, colIndices_)) {
    uint32 numRows = this->getNumRows();
    uint32 numCols = this->getNumCols();

    // Set column indices of the CSC matrix to zero...
    setArrayToZeros(colIndices_, numCols);

    // Determine the number of non-zero elements per column...
    for (uint32 i = 0; i < numRows; i++) {
        CsrLabelMatrix::index_const_iterator indexIterator = labelMatrix.row_indices_cbegin(i);
        uint32 numElements = labelMatrix.row_indices_cend(i) - indexIterator;

        for (uint32 j = 0; j < numElements; j++) {
            uint32 index = indexIterator[j];
            colIndices_[index]++;
        }
    }

    // Set the column indices of the CSC matrix with respect to the number of non-zero elements that correspond to
    // previous columns...
    uint32 tmp = 0;

    for (uint32 i = 0; i < numCols; i++) {
        uint32 index = colIndices_[i];
        colIndices_[i] = tmp;
        tmp += index;
    }

    // Set the row indices of the CSC matrix. This will modify the column indices...
    for (uint32 i = 0; i < numRows; i++) {
        CsrLabelMatrix::index_const_iterator indexIterator = labelMatrix.row_indices_cbegin(i);
        uint32 numElements = labelMatrix.row_indices_cend(i) - indexIterator;

        for (uint32 j = 0; j < numElements; j++) {
            uint32 originalIndex = indexIterator[j];
            uint32 index = colIndices_[originalIndex];
            rowIndices_[index] = i;
            colIndices_[originalIndex]++;
        }
    }

    // Reset the column indices to the previous values...
    tmp = 0;

    for (uint32 i = 0; i <= numCols; i++) {
        uint32 index = colIndices_[i];
        colIndices_[i] = tmp;
        tmp = index;
    }
}

CscLabelMatrix::~CscLabelMatrix() {
    free(rowIndices_);
    free(colIndices_);
}

CscLabelMatrix::index_const_iterator CscLabelMatrix::column_indices_cbegin(uint32 col) const {
    return view_.column_indices_cbegin(col);
}

CscLabelMatrix::index_const_iterator CscLabelMatrix::column_indices_cend(uint32 col) const {
    return view_.column_indices_cend(col);
}

CscLabelMatrix::value_const_iterator CscLabelMatrix::column_values_cbegin(uint32 col) const {
    return view_.column_values_cbegin(col);
}

CscLabelMatrix::value_const_iterator CscLabelMatrix::column_values_cend(uint32 col) const {
    return view_.column_values_cend(col);
}

uint32 CscLabelMatrix::getNumRows() const {
    return view_.getNumRows();
}

uint32 CscLabelMatrix::getNumCols() const {
    return view_.getNumCols();
}

uint32 CscLabelMatrix::getNumNonZeroElements() const {
    return view_.getNumNonZeroElements();
}
