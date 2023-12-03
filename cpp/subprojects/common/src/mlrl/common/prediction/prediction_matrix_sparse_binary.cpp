#include "mlrl/common/prediction/prediction_matrix_sparse_binary.hpp"

#include "mlrl/common/util/memory.hpp"

BinarySparsePredictionView::BinarySparsePredictionView(const BinaryLilMatrix& lilMatrix, uint32 numCols,
                                                       uint32 numNonZeroElements)
    : AllocatedBinaryCsrView(numNonZeroElements, lilMatrix.getNumRows(), numCols) {
    uint32 n = 0;

    for (uint32 i = 0; i < Matrix::numRows; i++) {
        BinarySparseMatrix::indptr[i] = n;

        for (auto it = lilMatrix.cbegin(i); it != lilMatrix.cend(i); it++) {
            BinarySparseMatrix::indices[n] = *it;
            n++;
        }
    }
}

BinarySparsePredictionView::BinarySparsePredictionView(BinarySparsePredictionView&& other)
    : AllocatedBinaryCsrView(std::move(other)) {}

BinarySparsePredictionMatrix::BinarySparsePredictionMatrix(const BinaryLilMatrix& lilMatrix, uint32 numCols,
                                                           uint32 numNonZeroElements)
    : IterableBinarySparseMatrixDecorator<MatrixDecorator<BinarySparsePredictionView>>(
      BinarySparsePredictionView(lilMatrix, numCols, numNonZeroElements)) {}

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
    return std::make_unique<BinarySparsePredictionMatrix>(lilMatrix, numCols, numNonZeroElements);
}
