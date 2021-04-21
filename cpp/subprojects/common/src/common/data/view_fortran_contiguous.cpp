#include "common/data/view_fortran_contiguous.hpp"


template<class T>
FortranContiguousConstView<T>::FortranContiguousConstView(uint32 numRows, uint32 numCols, T* array)
    : numRows_(numRows), numCols_(numCols), array_(array) {

}

template<class T>
typename FortranContiguousConstView<T>::const_iterator FortranContiguousConstView<T>::column_cbegin(uint32 col) const {
    return &array_[col * numRows_];
}

template<class T>
typename FortranContiguousConstView<T>::const_iterator FortranContiguousConstView<T>::column_cend(uint32 col) const {
    return &array_[(col + 1) * numRows_];
}

template<class T>
uint32 FortranContiguousConstView<T>::getNumRows() const {
    return numRows_;
}

template<class T>
uint32 FortranContiguousConstView<T>::getNumCols() const {
    return numCols_;
}

template class FortranContiguousConstView<uint8>;
template class FortranContiguousConstView<const uint8>;
template class FortranContiguousConstView<uint32>;
template class FortranContiguousConstView<const uint32>;
template class FortranContiguousConstView<float32>;
template class FortranContiguousConstView<const float32>;
template class FortranContiguousConstView<float64>;
template class FortranContiguousConstView<const float64>;

template<class T>
FortranContiguousView<T>::FortranContiguousView(uint32 numRows, uint32 numCols, T* array)
    : FortranContiguousConstView<T>(numRows, numCols, array) {

}

template<class T>
typename FortranContiguousView<T>::iterator FortranContiguousView<T>::column_begin(uint32 col) {
    return &FortranContiguousConstView<T>::array_[col * FortranContiguousConstView<T>::numRows_];
}

template<class T>
typename FortranContiguousView<T>::iterator FortranContiguousView<T>::column_end(uint32 col) {
    return &FortranContiguousConstView<T>::array_[(col + 1) * FortranContiguousConstView<T>::numRows_];
}

template class FortranContiguousView<uint8>;
template class FortranContiguousView<uint32>;
template class FortranContiguousView<float32>;
template class FortranContiguousView<float64>;
