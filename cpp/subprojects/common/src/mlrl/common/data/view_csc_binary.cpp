#include "mlrl/common/data/view_csc_binary.hpp"

BinaryCscView::BinaryCscView(uint32 numRows, uint32 numCols, uint32* rowIndices, uint32* indptr)
    : Matrix(numRows, numCols), rowIndices_(rowIndices), indptr_(indptr) {}

BinaryCscView::index_const_iterator BinaryCscView::indices_cbegin(uint32 col) const {
    return &rowIndices_[indptr_[col]];
}

BinaryCscView::index_const_iterator BinaryCscView::indices_cend(uint32 col) const {
    return &rowIndices_[indptr_[col + 1]];
}

BinaryCscView::index_iterator BinaryCscView::indices_begin(uint32 col) {
    return &rowIndices_[indptr_[col]];
}

BinaryCscView::index_iterator BinaryCscView::indices_end(uint32 col) {
    return &rowIndices_[indptr_[col + 1]];
}

uint32 BinaryCscView::getNumNonZeroElements() const {
    return indptr_[Matrix::numCols];
}
