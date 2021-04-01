#include "common/input/label_matrix_csr.hpp"


CsrLabelMatrix::CsrLabelMatrix(uint32 numRows, uint32 numCols, const uint32* rowIndices, const uint32* colIndices)
    : view_(BinaryCsrView(numRows, numCols, rowIndices, colIndices)) {

}

CsrLabelMatrix::index_const_iterator CsrLabelMatrix::row_indices_cbegin(uint32 row) const {
    return view_.row_indices_cbegin(row);
}

CsrLabelMatrix::index_const_iterator CsrLabelMatrix::row_indices_cend(uint32 row) const {
    return view_.row_indices_cend(row);
}

uint32 CsrLabelMatrix::getNumRows() const {
    return view_.getNumRows();
}

uint32 CsrLabelMatrix::getNumCols() const {
    return view_.getNumCols();
}
