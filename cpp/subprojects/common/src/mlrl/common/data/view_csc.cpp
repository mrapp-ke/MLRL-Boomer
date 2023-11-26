#include "mlrl/common/data/view_csc.hpp"

template<typename T>
CscView<T>::CscView(uint32 numRows, uint32 numCols, T* data, uint32* rowIndices, uint32* indptr)
    : numRows_(numRows), numCols_(numCols), data_(data), rowIndices_(rowIndices), indptr_(indptr) {}

template<typename T>
typename CscView<T>::index_const_iterator CscView<T>::indices_cbegin(uint32 col) const {
    return &rowIndices_[indptr_[col]];
}

template<typename T>
typename CscView<T>::index_const_iterator CscView<T>::indices_cend(uint32 col) const {
    return &rowIndices_[indptr_[col + 1]];
}

template<typename T>
typename CscView<T>::index_iterator CscView<T>::indices_begin(uint32 col) {
    return &rowIndices_[indptr_[col]];
}

template<typename T>
typename CscView<T>::index_iterator CscView<T>::indices_end(uint32 col) {
    return &rowIndices_[indptr_[col + 1]];
}

template<typename T>
typename CscView<T>::value_const_iterator CscView<T>::values_cbegin(uint32 col) const {
    return &data_[indptr_[col]];
}

template<typename T>
typename CscView<T>::value_const_iterator CscView<T>::values_cend(uint32 col) const {
    return &data_[indptr_[col + 1]];
}

template<typename T>
typename CscView<T>::value_iterator CscView<T>::values_begin(uint32 col) {
    return &data_[indptr_[col]];
}

template<typename T>
typename CscView<T>::value_iterator CscView<T>::values_end(uint32 col) {
    return &data_[indptr_[col + 1]];
}

template<typename T>
uint32 CscView<T>::getNumNonZeroElements() const {
    return indptr_[numCols_];
}

template<typename T>
uint32 CscView<T>::getNumRows() const {
    return numRows_;
}

template<typename T>
uint32 CscView<T>::getNumCols() const {
    return numCols_;
}

template class CscView<const uint8>;
template class CscView<uint8>;
template class CscView<const uint32>;
template class CscView<uint32>;
template class CscView<const int64>;
template class CscView<int64>;
template class CscView<const float32>;
template class CscView<float32>;
template class CscView<const float64>;
template class CscView<float64>;
