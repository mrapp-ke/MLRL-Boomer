#include "common/data/view_csc_binary.hpp"

BinaryCscConstView::BinaryCscConstView(uint32 numRows, uint32 numCols, uint32* rowIndices, uint32* indptr)
    : numRows_(numRows), numCols_(numCols), rowIndices_(rowIndices), indptr_(indptr) {}

BinaryCscConstView::index_const_iterator BinaryCscConstView::indices_cbegin(uint32 col) const {
    return &rowIndices_[indptr_[col]];
}

BinaryCscConstView::index_const_iterator BinaryCscConstView::indices_cend(uint32 col) const {
    return &rowIndices_[indptr_[col + 1]];
}

uint32 BinaryCscConstView::getNumRows() const {
    return numRows_;
}

uint32 BinaryCscConstView::getNumCols() const {
    return numCols_;
}

uint32 BinaryCscConstView::getNumNonZeroElements() const {
    return indptr_[numCols_];
}

BinaryCscView::BinaryCscView(uint32 numRows, uint32 numCols, uint32* rowIndices, uint32* indptr)
    : BinaryCscConstView(numRows, numCols, rowIndices, indptr) {}

BinaryCscView::index_iterator BinaryCscView::indices_begin(uint32 col) {
    return &BinaryCscConstView::rowIndices_[BinaryCscConstView::indptr_[col]];
}

BinaryCscView::index_iterator BinaryCscView::indices_end(uint32 col) {
    return &BinaryCscConstView::rowIndices_[BinaryCscConstView::indptr_[col + 1]];
}
