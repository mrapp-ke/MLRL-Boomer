#include "mlrl/common/data/view_csr_binary.hpp"

BinaryCsrView::BinaryCsrView(uint32 numRows, uint32 numCols, uint32* colIndices, uint32* indptr)
    : numRows_(numRows), numCols_(numCols), colIndices_(colIndices), indptr_(indptr) {}

BinaryCsrView::index_const_iterator BinaryCsrView::indices_cbegin(uint32 row) const {
    return &colIndices_[indptr_[row]];
}

BinaryCsrView::index_const_iterator BinaryCsrView::indices_cend(uint32 row) const {
    return &colIndices_[indptr_[row + 1]];
}

BinaryCsrView::index_iterator BinaryCsrView::indices_begin(uint32 row) {
    return &colIndices_[indptr_[row]];
}

BinaryCsrView::index_iterator BinaryCsrView::indices_end(uint32 row) {
    return &colIndices_[indptr_[row + 1]];
}

uint32 BinaryCsrView::getNumNonZeroElements() const {
    return indptr_[numRows_];
}

uint32 BinaryCsrView::getNumRows() const {
    return numRows_;
}

uint32 BinaryCsrView::getNumCols() const {
    return numCols_;
}
