#include "mlrl/common/data/view_csr.hpp"

template<typename T>
CsrView<T>::CsrView(uint32 numRows, uint32 numCols, T* data, uint32* colIndices, uint32* indptr)
    : Matrix(numRows, numCols), data_(data), colIndices_(colIndices), indptr_(indptr) {}

template<typename T>
typename CsrView<T>::index_const_iterator CsrView<T>::indices_cbegin(uint32 row) const {
    return &colIndices_[indptr_[row]];
}

template<typename T>
typename CsrView<T>::index_const_iterator CsrView<T>::indices_cend(uint32 row) const {
    return &colIndices_[indptr_[row + 1]];
}

template<typename T>
typename CsrView<T>::index_iterator CsrView<T>::indices_begin(uint32 row) {
    return &colIndices_[indptr_[row]];
}

template<typename T>
typename CsrView<T>::index_iterator CsrView<T>::indices_end(uint32 row) {
    return &colIndices_[indptr_[row + 1]];
}

template<typename T>
typename CsrView<T>::value_const_iterator CsrView<T>::values_cbegin(uint32 row) const {
    return &data_[indptr_[row]];
}

template<typename T>
typename CsrView<T>::value_const_iterator CsrView<T>::values_cend(uint32 row) const {
    return &data_[indptr_[row + 1]];
}

template<typename T>
typename CsrView<T>::value_iterator CsrView<T>::values_begin(uint32 row) {
    return &data_[indptr_[row]];
}

template<typename T>
typename CsrView<T>::value_iterator CsrView<T>::values_end(uint32 row) {
    return &data_[indptr_[row + 1]];
}

template<typename T>
uint32 CsrView<T>::getNumNonZeroElements() const {
    return indptr_[Matrix::numCols];
}

template class CsrView<const uint8>;
template class CsrView<uint8>;
template class CsrView<const uint32>;
template class CsrView<uint32>;
template class CsrView<const int64>;
template class CsrView<int64>;
template class CsrView<const float32>;
template class CsrView<float32>;
template class CsrView<const float64>;
template class CsrView<float64>;
