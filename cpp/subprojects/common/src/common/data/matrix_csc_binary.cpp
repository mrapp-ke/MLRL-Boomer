#include "common/data/matrix_csc_binary.hpp"

#include <cstdlib>

BinaryCscMatrix::BinaryCscMatrix(uint32 numRows, uint32 numCols, uint32 numNonZeroElements)
    : BinaryCscView(numRows, numCols, (uint32*) malloc(numNonZeroElements * sizeof(uint32)),
                    (uint32*) malloc((numCols + 1) * sizeof(uint32))) {}

BinaryCscMatrix::~BinaryCscMatrix() {
    free(rowIndices_);
    free(indptr_);
}
