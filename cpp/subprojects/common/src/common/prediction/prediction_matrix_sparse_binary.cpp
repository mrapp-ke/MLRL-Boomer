#include "common/prediction/prediction_matrix_sparse_binary.hpp"

#include <cstdlib>

BinarySparsePredictionMatrix::BinarySparsePredictionMatrix(uint32 numRows, uint32 numCols, uint32* colIndices,
                                                           uint32* indptr)
    : BinaryCsrConstView(numRows, numCols, colIndices, indptr), colIndices_(colIndices), indptr_(indptr) {}

BinarySparsePredictionMatrix::~BinarySparsePredictionMatrix() {
    free(colIndices_);
    free(indptr_);
}

uint32* BinarySparsePredictionMatrix::getColIndices() {
    return colIndices_;
}

uint32* BinarySparsePredictionMatrix::releaseColIndices() {
    uint32* ptr = colIndices_;
    colIndices_ = nullptr;
    return ptr;
}

uint32* BinarySparsePredictionMatrix::getIndptr() {
    return indptr_;
}

uint32* BinarySparsePredictionMatrix::releaseIndptr() {
    uint32* ptr = indptr_;
    indptr_ = nullptr;
    return ptr;
}

std::unique_ptr<BinarySparsePredictionMatrix> createBinarySparsePredictionMatrix(const BinaryLilMatrix& lilMatrix,
                                                                                 uint32 numCols,
                                                                                 uint32 numNonZeroElements) {
    uint32 numRows = lilMatrix.getNumRows();
    uint32* colIndices = (uint32*) malloc(numNonZeroElements * sizeof(uint32));
    uint32* indptr = (uint32*) malloc((numRows + 1) * sizeof(uint32));
    uint32 n = 0;

    for (uint32 i = 0; i < numRows; i++) {
        indptr[i] = n;

        for (auto it = lilMatrix.cbegin(i); it != lilMatrix.cend(i); it++) {
            colIndices[n] = *it;
            n++;
        }
    }

    indptr[numRows] = n;
    return std::make_unique<BinarySparsePredictionMatrix>(numRows, numCols, colIndices, indptr);
}
