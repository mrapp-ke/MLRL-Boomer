#include "common/data/view_csr_binary.hpp"


BinaryCsrView::BinaryCsrView(uint32 numRows, uint32 numCols, const uint32* rowIndices, const uint32* colIndices)
    : numRows_(numRows), numCols_(numCols), rowIndices_(rowIndices), colIndices_(colIndices) {

}

BinaryCsrView::index_const_iterator BinaryCsrView::row_indices_cbegin(uint32 row) const {
    return &colIndices_[rowIndices_[row]];
}

BinaryCsrView::index_const_iterator BinaryCsrView::row_indices_cend(uint32 row) const {
    return &colIndices_[rowIndices_[row + 1]];
}

uint32 BinaryCsrView::getNumRows() const {
    return numRows_;
}

uint32 BinaryCsrView::getNumCols() const {
    return numCols_;
}
