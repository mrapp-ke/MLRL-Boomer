#include "mlrl/common/prediction/prediction_matrix_sparse_binary.hpp"

#include "mlrl/common/util/memory.hpp"

BinarySparsePredictionMatrix::BinarySparsePredictionMatrix(uint32* indices, uint32* indptr, uint32 numRows,
                                                           uint32 numCols)
    : IterableBinarySparseMatrixDecorator<MatrixDecorator<BinaryCsrView>>(
      BinaryCsrView(indices, indptr, numRows, numCols)) {}

BinarySparsePredictionMatrix::~BinarySparsePredictionMatrix() {
    freeMemory(this->view.indices);
    freeMemory(this->view.indptr);
}

uint32* BinarySparsePredictionMatrix::getIndices() {
    return this->view.indices;
}

uint32* BinarySparsePredictionMatrix::releaseIndices() {
    return this->view.releaseIndices();
}

uint32* BinarySparsePredictionMatrix::getIndptr() {
    return this->view.indptr;
}

uint32* BinarySparsePredictionMatrix::releaseIndptr() {
    return this->view.releaseIndptr();
}

std::unique_ptr<BinarySparsePredictionMatrix> createBinarySparsePredictionMatrix(const BinaryLilMatrix& lilMatrix,
                                                                                 uint32 numCols,
                                                                                 uint32 numNonZeroElements) {
    uint32 numRows = lilMatrix.getNumRows();
    uint32* indices = allocateMemory<uint32>(numNonZeroElements);
    uint32* indptr = allocateMemory<uint32>(numRows + 1);
    uint32 n = 0;

    for (uint32 i = 0; i < numRows; i++) {
        indptr[i] = n;

        for (auto it = lilMatrix.cbegin(i); it != lilMatrix.cend(i); it++) {
            indices[n] = *it;
            n++;
        }
    }

    indptr[numRows] = n;
    return std::make_unique<BinarySparsePredictionMatrix>(indices, indptr, numRows, numCols);
}
