#include "common/data/view_csr.hpp"


template<class T>
CsrConstView<T>::CsrConstView(uint32 numRows, uint32 numCols, T* data, uint32* rowIndices, uint32* colIndices)
    : numRows_(numRows), numCols_(numCols), data_(data), rowIndices_(rowIndices), colIndices_(colIndices) {

}

template<class T>
typename CsrConstView<T>::value_const_iterator CsrConstView<T>::row_values_cbegin(uint32 row) const {
    return &data_[rowIndices_[row]];
}

template<class T>
typename CsrConstView<T>::value_const_iterator CsrConstView<T>::row_values_cend(uint32 row) const {
    return &data_[rowIndices_[row + 1]];
}

template<class T>
typename CsrConstView<T>::index_const_iterator CsrConstView<T>::row_indices_cbegin(uint32 row) const {
    return &colIndices_[rowIndices_[row]];
}

template<class T>
typename CsrConstView<T>::index_const_iterator CsrConstView<T>::row_indices_cend(uint32 row) const {
    return &colIndices_[rowIndices_[row + 1]];
}

template<class T>
uint32 CsrConstView<T>::getNumRows() const {
    return numRows_;
}

template<class T>
uint32 CsrConstView<T>::getNumCols() const {
    return numCols_;
}

template<class T>
uint32 CsrConstView<T>::getNumNonZeroElements() const {
    return rowIndices_[numCols_];
}

template class CsrConstView<uint8>;
template class CsrConstView<const uint8>;
template class CsrConstView<uint32>;
template class CsrConstView<const uint32>;
template class CsrConstView<float32>;
template class CsrConstView<const float32>;
template class CsrConstView<float64>;
template class CsrConstView<const float64>;

template<class T>
CsrView<T>::CsrView(uint32 numRows, uint32 numCols, T* data, uint32* rowIndices, uint32* colIndices)
    : CsrConstView<T>(numRows, numCols, data, rowIndices, colIndices) {

}

template<class T>
typename CsrView<T>::value_iterator CsrView<T>::row_values_begin(uint32 row) {
    return &CsrConstView<T>::data_[CsrConstView<T>::rowIndices_[row]];
}

template<class T>
typename CsrView<T>::value_iterator CsrView<T>::row_values_end(uint32 row) {
    return &CsrConstView<T>::data_[CsrConstView<T>::rowIndices_[row + 1]];
}

template<class T>
typename CsrView<T>::index_iterator CsrView<T>::row_indices_begin(uint32 row) {
    return &CsrConstView<T>::colIndices_[CsrConstView<T>::rowIndices_[row]];
}

template<class T>
typename CsrView<T>::index_iterator CsrView<T>::row_indices_end(uint32 row) {
    return &CsrConstView<T>::colIndices_[CsrConstView<T>::rowIndices_[row + 1]];
}

template class CsrView<uint8>;
template class CsrView<uint32>;
template class CsrView<float32>;
template class CsrView<float64>;
