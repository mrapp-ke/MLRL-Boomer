#include "mlrl/common/data/matrix_sparse_set.hpp"

#include "mlrl/common/data/triple.hpp"
#include "mlrl/common/data/tuple.hpp"

template<typename T>
SparseSetMatrix<T>::SparseSetMatrix(uint32 numRows, uint32 numCols)
    : MatrixDecorator<AllocatedSparseSetView<T>>(AllocatedSparseSetView<T>(numRows, numCols)) {}

template<typename T>
typename SparseSetMatrix<T>::iterator SparseSetMatrix<T>::begin(uint32 row) {
    return this->view.begin(row);
}

template<typename T>
typename SparseSetMatrix<T>::iterator SparseSetMatrix<T>::end(uint32 row) {
    return this->view.end(row);
}

template<typename T>
typename SparseSetMatrix<T>::const_iterator SparseSetMatrix<T>::cbegin(uint32 row) const {
    return this->view.cbegin(row);
}

template<typename T>
typename SparseSetMatrix<T>::const_iterator SparseSetMatrix<T>::cend(uint32 row) const {
    return this->view.cend(row);
}

template<typename T>
typename SparseSetMatrix<T>::row SparseSetMatrix<T>::operator[](uint32 row) {
    return this->view[row];
}

template<typename T>
typename SparseSetMatrix<T>::const_row SparseSetMatrix<T>::operator[](uint32 row) const {
    return this->view[row];
}

template<typename T>
uint32 SparseSetMatrix<T>::getNumRows() const {
    return this->view.numRows;
}

template<typename T>
uint32 SparseSetMatrix<T>::getNumCols() const {
    return this->view.numCols;
}

template<typename T>
void SparseSetMatrix<T>::clear() {
    uint32 numRows = this->view.numRows;

    for (uint32 i = 0; i < numRows; i++) {
        this->view[i].clear();
    }
}

template class SparseSetMatrix<uint8>;
template class SparseSetMatrix<uint32>;
template class SparseSetMatrix<int64>;
template class SparseSetMatrix<float32>;
template class SparseSetMatrix<float64>;
template class SparseSetMatrix<Tuple<uint8>>;
template class SparseSetMatrix<Tuple<uint32>>;
template class SparseSetMatrix<Tuple<int64>>;
template class SparseSetMatrix<Tuple<float32>>;
template class SparseSetMatrix<Tuple<float64>>;
template class SparseSetMatrix<Triple<uint8>>;
template class SparseSetMatrix<Triple<uint32>>;
template class SparseSetMatrix<Triple<int64>>;
template class SparseSetMatrix<Triple<float32>>;
template class SparseSetMatrix<Triple<float64>>;
