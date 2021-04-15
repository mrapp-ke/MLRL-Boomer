#include "common/data/view_csc_binary.hpp"


BinaryCscView::BinaryCscView(uint32 numRows, uint32 numCols, uint32* rowIndices, uint32* colIndices)
    : numRows_(numRows), numCols_(numCols), rowIndices_(rowIndices), colIndices_(colIndices) {

}

BinaryCscView::index_iterator BinaryCscView::column_indices_begin(uint32 col) {
    return &rowIndices_[colIndices_[col]];
}

BinaryCscView::index_iterator BinaryCscView::column_indices_end(uint32 col) {
    return &rowIndices_[colIndices_[col + 1]];
}

BinaryCscView::index_const_iterator BinaryCscView::column_indices_cbegin(uint32 col) const {
    return &rowIndices_[colIndices_[col]];
}

BinaryCscView::index_const_iterator BinaryCscView::column_indices_cend(uint32 col) const {
    return &rowIndices_[colIndices_[col + 1]];
}

BinaryCscView::value_const_iterator BinaryCscView::column_values_cbegin(uint32 col) const {
    return make_index_forward_iterator(this->column_indices_cbegin(col), this->column_indices_cend(col));
}

BinaryCscView::value_const_iterator BinaryCscView::column_values_cend(uint32 col) const {
    return make_index_forward_iterator(this->column_indices_cbegin(col), this->column_indices_cend(col), numRows_);
}

uint32 BinaryCscView::getNumRows() const {
    return numRows_;
}

uint32 BinaryCscView::getNumCols() const {
    return numCols_;
}

uint32 BinaryCscView::getNumNonZeroElements() const {
    return colIndices_[numCols_];
}
