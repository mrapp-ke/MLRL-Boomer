#include "common/data/view_csr_binary.hpp"


BinaryCsrView::BinaryCsrView(uint32 numRows, uint32 numCols, uint32* rowIndices, uint32* colIndices)
    : numRows_(numRows), numCols_(numCols), rowIndices_(rowIndices), colIndices_(colIndices) {

}

BinaryCsrView::index_iterator BinaryCsrView::row_indices_begin(uint32 row) {
    return &colIndices_[rowIndices_[row]];
}

BinaryCsrView::index_iterator BinaryCsrView::row_indices_end(uint32 row) {
    return &colIndices_[rowIndices_[row + 1]];
}

BinaryCsrView::index_const_iterator BinaryCsrView::row_indices_cbegin(uint32 row) const {
    return &colIndices_[rowIndices_[row]];
}

BinaryCsrView::index_const_iterator BinaryCsrView::row_indices_cend(uint32 row) const {
    return &colIndices_[rowIndices_[row + 1]];
}

BinaryCsrView::value_const_iterator BinaryCsrView::row_values_cbegin(uint32 row) const {
    return make_index_forward_iterator(this->row_indices_cbegin(row), this->row_indices_cend(row));
}

BinaryCsrView::value_const_iterator BinaryCsrView::row_values_cend(uint32 row) const {
    return make_index_forward_iterator(this->row_indices_cbegin(row), this->row_indices_cend(row), numCols_);
}

uint32 BinaryCsrView::getNumRows() const {
    return numRows_;
}

uint32 BinaryCsrView::getNumCols() const {
    return numCols_;
}

uint32 BinaryCsrView::getNumNonZeroElements() const {
    return rowIndices_[numRows_];
}
