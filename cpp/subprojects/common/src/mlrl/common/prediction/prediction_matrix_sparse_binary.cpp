#include "mlrl/common/prediction/prediction_matrix_sparse_binary.hpp"

#include "mlrl/common/util/memory.hpp"

BinarySparsePredictionMatrix::BinarySparsePredictionMatrix(uint32 numRows, uint32 numCols, uint32* colIndices,
                                                           uint32* indptr)
    : BinaryCsrView(numRows, numCols, colIndices, indptr), colIndices_(colIndices), indptr_(indptr) {}

BinarySparsePredictionMatrix::~BinarySparsePredictionMatrix() {
    freeMemory(colIndices_);
    freeMemory(indptr_);
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
    uint32* colIndices = allocateMemory<uint32>(numNonZeroElements);
    uint32* indptr = allocateMemory<uint32>(numRows + 1);
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
