#include "common/output/prediction_matrix_sparse_binary.hpp"
#include <cstdlib>


BinarySparsePredictionMatrix::BinarySparsePredictionMatrix(const BinaryLilMatrix& lilMatrix, uint32 numCols,
                                                           uint32 numNonZeroElements)
    : rowIndices_((uint32*) malloc((lilMatrix.getNumRows() + 1) * sizeof(uint32))),
      colIndices_((uint32*) malloc(numNonZeroElements * sizeof(uint32))),
      view_(BinaryCsrConstView(lilMatrix.getNumRows(), numCols, rowIndices_, colIndices_)) {
    uint32 numRows = lilMatrix.getNumRows();
    uint32 n = 0;

    for (uint32 i = 0; i < numRows; i++) {
        rowIndices_[i] = n;

        for (auto it = lilMatrix.row_cbegin(i); it != lilMatrix.row_cend(i); it++) {
            colIndices_[n] = *it;
            n++;
        }
    }

    rowIndices_[numRows] = n;
}

BinarySparsePredictionMatrix::~BinarySparsePredictionMatrix() {
    free(rowIndices_);
    free(colIndices_);
}

BinarySparsePredictionMatrix::index_const_iterator BinarySparsePredictionMatrix::row_indices_cbegin(uint32 row) const {
    return view_.row_indices_cbegin(row);
}

BinarySparsePredictionMatrix::index_const_iterator BinarySparsePredictionMatrix::row_indices_cend(uint32 row) const {
    return view_.row_indices_cend(row);
}

BinarySparsePredictionMatrix::value_const_iterator BinarySparsePredictionMatrix::row_values_cbegin(uint32 row) const {
    return view_.row_values_cbegin(row);
}

BinarySparsePredictionMatrix::value_const_iterator BinarySparsePredictionMatrix::row_values_cend(uint32 row) const {
    return view_.row_values_cend(row);
}

uint32 BinarySparsePredictionMatrix::getNumRows() const {
    return view_.getNumRows();
}

uint32 BinarySparsePredictionMatrix::getNumCols() const {
    return view_.getNumCols();
}

uint32 BinarySparsePredictionMatrix::getNumNonZeroElements() const {
    return view_.getNumNonZeroElements();
}

uint32* BinarySparsePredictionMatrix::releaseRowIndices() {
    uint32* ptr = rowIndices_;
    rowIndices_ = nullptr;
    return ptr;
}

uint32* BinarySparsePredictionMatrix::releaseColIndices() {
    uint32* ptr = colIndices_;
    colIndices_ = nullptr;
    return ptr;
}
