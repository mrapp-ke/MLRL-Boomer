#include "common/data/view_csr.hpp"


template<class T>
CsrView<T>::CsrView(uint32 numRows, uint32 numCols, const T* data, const uint32* rowIndices, const uint32* colIndices)
    : BinaryCsrView(numRows, numCols, rowIndices, colIndices), data_(data) {

}

template<class T>
typename CsrView<T>::value_const_iterator CsrView<T>::row_values_cbegin(uint32 row) const {
    return &data_[rowIndices_[row]];
}

template<class T>
typename CsrView<T>::value_const_iterator CsrView<T>::row_values_cend(uint32 row) const {
    return &data_[rowIndices_[row + 1]];
}

template class CsrView<float32>;
