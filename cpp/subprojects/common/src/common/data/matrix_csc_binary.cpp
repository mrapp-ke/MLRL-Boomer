#include "common/data/matrix_csc_binary.hpp"

#include <cstdlib>

BinaryCscMatrix::BinaryCscMatrix(uint32 numRows, uint32 numCols, uint32 numNonZeroElements)
    : BinaryCscView(numRows, numCols, (uint32*) malloc(numNonZeroElements * sizeof(uint32)),
                    (uint32*) malloc((numCols + 1) * sizeof(uint32))),
      maxCapacity_(numNonZeroElements) {
    indptr_[numCols_] = numNonZeroElements;
}

BinaryCscMatrix::~BinaryCscMatrix() {
    free(rowIndices_);
    free(indptr_);
}

BinaryCscMatrix::index_const_iterator BinaryCscMatrix::indptr_cbegin() const {
    return indptr_;
}

BinaryCscMatrix::index_const_iterator BinaryCscMatrix::indptr_cend() const {
    return &indptr_[numCols_];
}

BinaryCscMatrix::index_iterator BinaryCscMatrix::indptr_begin() {
    return indptr_;
}

BinaryCscMatrix::index_iterator BinaryCscMatrix::indptr_end() {
    return &indptr_[numCols_];
}

void BinaryCscMatrix::setNumNonZeroElements(uint32 numNonZeroElements, bool freeMemory) {
    if (numNonZeroElements < maxCapacity_) {
        if (freeMemory) {
            rowIndices_ = (uint32*) realloc(rowIndices_, numNonZeroElements * sizeof(uint32));
            maxCapacity_ = numNonZeroElements;
        }
    } else if (numNonZeroElements > maxCapacity_) {
        rowIndices_ = (uint32*) realloc(rowIndices_, numNonZeroElements * sizeof(uint32));
        maxCapacity_ = numNonZeroElements;
    }

    indptr_[numCols_] = numNonZeroElements;
}
