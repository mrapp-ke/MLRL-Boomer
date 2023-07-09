#include "common/data/matrix_csc_binary.hpp"

#include <cstdlib>

BinaryCscMatrix::BinaryCscMatrix(uint32 numRows, uint32 numCols, uint32 numNonZeroElements)
    : BinaryCscView(numRows, numCols, (uint32*) malloc(numNonZeroElements * sizeof(uint32)),
                    (uint32*) malloc((numCols + 1) * sizeof(uint32))) {}

BinaryCscMatrix::~BinaryCscMatrix() {
    free(rowIndices_);
    free(indptr_);
}

BinaryCscMatrix::index_const_iterator BinaryCscMatrix::indptr_cbegin() const {
    return indptr_;
}

BinaryCscMatrix::index_const_iterator BinaryCscMatrix::indptr_cend() const {
    return &indptr_[numCols_ + 1];
}

BinaryCscMatrix::index_iterator BinaryCscMatrix::indptr_begin() {
    return indptr_;
}

BinaryCscMatrix::index_iterator BinaryCscMatrix::indptr_end() {
    return &indptr_[numCols_ + 1];
}
