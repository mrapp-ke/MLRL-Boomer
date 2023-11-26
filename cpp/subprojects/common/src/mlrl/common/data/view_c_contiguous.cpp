#include "mlrl/common/data/view_c_contiguous.hpp"

template<typename T>
CContiguousView<T>::CContiguousView(uint32 numRows, uint32 numCols, T* array)
    : numRows_(numRows), numCols_(numCols), array_(array) {}

template<typename T>
typename CContiguousView<T>::value_const_iterator CContiguousView<T>::values_cbegin(uint32 row) const {
    return &array_[row * numCols_];
}

template<typename T>
typename CContiguousView<T>::value_const_iterator CContiguousView<T>::values_cend(uint32 row) const {
    return &array_[(row + 1) * numCols_];
}

template<typename T>
typename CContiguousView<T>::value_iterator CContiguousView<T>::values_begin(uint32 row) {
    return &array_[row * numCols_];
}

template<typename T>
typename CContiguousView<T>::value_iterator CContiguousView<T>::values_end(uint32 row) {
    return &array_[(row + 1) * numCols_];
}

template<typename T>
uint32 CContiguousView<T>::getNumRows() const {
    return numRows_;
}

template<typename T>
uint32 CContiguousView<T>::getNumCols() const {
    return numCols_;
}

template class CContiguousView<const uint8>;
template class CContiguousView<uint8>;
template class CContiguousView<const uint32>;
template class CContiguousView<uint32>;
template class CContiguousView<const int64>;
template class CContiguousView<int64>;
template class CContiguousView<const float32>;
template class CContiguousView<float32>;
template class CContiguousView<const float64>;
template class CContiguousView<float64>;
