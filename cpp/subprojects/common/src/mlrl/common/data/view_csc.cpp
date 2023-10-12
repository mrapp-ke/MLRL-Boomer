#include "mlrl/common/data/view_csc.hpp"

template<typename T>
CscConstView<T>::CscConstView(uint32 numRows, uint32 numCols, T* data, uint32* rowIndices, uint32* indptr)
    : numRows_(numRows), numCols_(numCols), data_(data), rowIndices_(rowIndices), indptr_(indptr) {}

template<typename T>
typename CscConstView<T>::value_const_iterator CscConstView<T>::values_cbegin(uint32 col) const {
    return &data_[indptr_[col]];
}

template<typename T>
typename CscConstView<T>::value_const_iterator CscConstView<T>::values_cend(uint32 col) const {
    return &data_[indptr_[col + 1]];
}

template<typename T>
typename CscConstView<T>::index_const_iterator CscConstView<T>::indices_cbegin(uint32 col) const {
    return &rowIndices_[indptr_[col]];
}

template<typename T>
typename CscConstView<T>::index_const_iterator CscConstView<T>::indices_cend(uint32 col) const {
    return &rowIndices_[indptr_[col + 1]];
}

template<typename T>
uint32 CscConstView<T>::getNumNonZeroElements() const {
    return indptr_[numCols_];
}

template<typename T>
uint32 CscConstView<T>::getNumRows() const {
    return numRows_;
}

template<typename T>
uint32 CscConstView<T>::getNumCols() const {
    return numCols_;
}

template class CscConstView<uint8>;
template class CscConstView<const uint8>;
template class CscConstView<uint32>;
template class CscConstView<const uint32>;
template class CscConstView<int64>;
template class CscConstView<const int64>;
template class CscConstView<float32>;
template class CscConstView<const float32>;
template class CscConstView<float64>;
template class CscConstView<const float64>;

template<typename T>
CscView<T>::CscView(uint32 numRows, uint32 numCols, T* data, uint32* rowIndices, uint32* indptr)
    : CscConstView<T>(numRows, numCols, data, rowIndices, indptr) {}

template<typename T>
typename CscView<T>::value_iterator CscView<T>::values_begin(uint32 col) {
    return &CscConstView<T>::data_[CscConstView<T>::indptr_[col]];
}

template<typename T>
typename CscView<T>::value_iterator CscView<T>::values_end(uint32 col) {
    return &CscConstView<T>::data_[CscConstView<T>::indptr_[col + 1]];
}

template<typename T>
typename CscView<T>::index_iterator CscView<T>::indices_begin(uint32 col) {
    return &CscConstView<T>::rowIndices_[CscConstView<T>::indptr_[col]];
}

template<typename T>
typename CscView<T>::index_iterator CscView<T>::indices_end(uint32 col) {
    return &CscConstView<T>::rowIndices_[CscConstView<T>::indptr_[col + 1]];
}

template class CscView<uint8>;
template class CscView<uint32>;
template class CscView<int64>;
template class CscView<float32>;
template class CscView<float64>;
