#include "mlrl/common/data/view_fortran_contiguous.hpp"

template<typename T>
FortranContiguousView<T>::FortranContiguousView(uint32 numRows, uint32 numCols, T* array)
    : numRows_(numRows), numCols_(numCols), array_(array) {}

template<typename T>
typename FortranContiguousView<T>::value_const_iterator FortranContiguousView<T>::values_cbegin(uint32 col) const {
    return &array_[col * numRows_];
}

template<typename T>
typename FortranContiguousView<T>::value_const_iterator FortranContiguousView<T>::values_cend(uint32 col) const {
    return &array_[(col + 1) * numRows_];
}

template<typename T>
typename FortranContiguousView<T>::value_iterator FortranContiguousView<T>::values_begin(uint32 col) {
    return &array_[col * numRows_];
}

template<typename T>
typename FortranContiguousView<T>::value_iterator FortranContiguousView<T>::values_end(uint32 col) {
    return &array_[(col + 1) * numRows_];
}

template<typename T>
uint32 FortranContiguousView<T>::getNumRows() const {
    return numRows_;
}

template<typename T>
uint32 FortranContiguousView<T>::getNumCols() const {
    return numCols_;
}

template class FortranContiguousView<const uint8>;
template class FortranContiguousView<uint8>;
template class FortranContiguousView<const uint32>;
template class FortranContiguousView<uint32>;
template class FortranContiguousView<const int64>;
template class FortranContiguousView<int64>;
template class FortranContiguousView<const float32>;
template class FortranContiguousView<float32>;
template class FortranContiguousView<const float64>;
template class FortranContiguousView<float64>;
