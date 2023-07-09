#include "common/data/view_csr_binary.hpp"

BinaryCsrConstView::BinaryCsrConstView(uint32 numRows, uint32 numCols, uint32* colIndices, uint32* indptr)
    : numRows_(numRows), numCols_(numCols), colIndices_(colIndices), indptr_(indptr) {}

BinaryCsrConstView::index_const_iterator BinaryCsrConstView::indices_cbegin(uint32 row) const {
    return &colIndices_[indptr_[row]];
}

BinaryCsrConstView::index_const_iterator BinaryCsrConstView::indices_cend(uint32 row) const {
    return &colIndices_[indptr_[row + 1]];
}

uint32 BinaryCsrConstView::getNumNonZeroElements() const {
    return indptr_[numRows_];
}

uint32 BinaryCsrConstView::getNumRows() const {
    return numRows_;
}

uint32 BinaryCsrConstView::getNumCols() const {
    return numCols_;
}

BinaryCsrView::BinaryCsrView(uint32 numRows, uint32 numCols, uint32* colIndices, uint32* indptr)
    : BinaryCsrConstView(numRows, numCols, colIndices, indptr) {}

BinaryCsrView::index_iterator BinaryCsrView::indices_begin(uint32 row) {
    return &BinaryCsrConstView::colIndices_[BinaryCsrConstView::indptr_[row]];
}

BinaryCsrView::index_iterator BinaryCsrView::indices_end(uint32 row) {
    return &BinaryCsrConstView::colIndices_[BinaryCsrConstView::indptr_[row + 1]];
}
