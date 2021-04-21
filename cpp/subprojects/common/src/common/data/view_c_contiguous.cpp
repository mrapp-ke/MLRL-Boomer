#include "common/data/view_c_contiguous.hpp"


template<class T>
CContiguousConstView<T>::CContiguousConstView(uint32 numRows, uint32 numCols, T* array)
    : numRows_(numRows), numCols_(numCols), array_(array) {

}

template<class T>
typename CContiguousConstView<T>::const_iterator CContiguousConstView<T>::row_cbegin(uint32 row) const {
    return &array_[row * numCols_];
}

template<class T>
typename CContiguousConstView<T>::const_iterator CContiguousConstView<T>::row_cend(uint32 row) const {
    return &array_[(row + 1) * numCols_];
}

template<class T>
uint32 CContiguousConstView<T>::getNumRows() const {
    return numRows_;
}

template<class T>
uint32 CContiguousConstView<T>::getNumCols() const {
    return numCols_;
}

template class CContiguousConstView<uint8>;
template class CContiguousConstView<const uint8>;
template class CContiguousConstView<uint32>;
template class CContiguousConstView<const uint32>;
template class CContiguousConstView<float32>;
template class CContiguousConstView<const float32>;
template class CContiguousConstView<float64>;
template class CContiguousConstView<const float64>;

template<class T>
CContiguousView<T>::CContiguousView(uint32 numRows, uint32 numCols, T* array)
    : CContiguousConstView<T>(numRows, numCols, array) {

}

template<class T>
typename CContiguousView<T>::iterator CContiguousView<T>::row_begin(uint32 row) {
    return &CContiguousConstView<T>::array_[row * CContiguousConstView<T>::numCols_];
}

template<class T>
typename CContiguousView<T>::iterator CContiguousView<T>::row_end(uint32 row) {
    return &CContiguousConstView<T>::array_[(row + 1) * CContiguousConstView<T>::numCols_];
}

template class CContiguousView<uint8>;
template class CContiguousView<uint32>;
template class CContiguousView<float32>;
template class CContiguousView<float64>;
