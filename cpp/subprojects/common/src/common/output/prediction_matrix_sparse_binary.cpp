#include "common/output/prediction_matrix_sparse_binary.hpp"
#include <cstdlib>


BinarySparsePredictionMatrix::BinarySparsePredictionMatrix(uint32 numRows, uint32 numCols, uint32* rowIndices,
                                                           uint32* colIndices)
    : rowIndices_(rowIndices), colIndices_(colIndices),
      view_(BinaryCsrConstView(numRows, numCols, rowIndices_, colIndices_)) {

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

std::unique_ptr<BinarySparsePredictionMatrix> createBinarySparsePredictionMatrix(const BinaryLilMatrix& lilMatrix,
                                                                                 uint32 numCols,
                                                                                 uint32 numNonZeroElements) {
    uint32 numRows = lilMatrix.getNumRows();
    uint32* rowIndices = (uint32*) malloc((numRows + 1) * sizeof(uint32));
    uint32* colIndices = (uint32*) malloc(numNonZeroElements * sizeof(uint32));
    uint32 n = 0;

    for (uint32 i = 0; i < numRows; i++) {
        rowIndices[i] = n;

        for (auto it = lilMatrix.row_cbegin(i); it != lilMatrix.row_cend(i); it++) {
            colIndices[n] = *it;
            n++;
        }
    }

    rowIndices[numRows] = n;
    return std::make_unique<BinarySparsePredictionMatrix>(numRows, numCols, rowIndices, colIndices);
}
