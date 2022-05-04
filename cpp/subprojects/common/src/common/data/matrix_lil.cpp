#include "common/data/matrix_lil.hpp"
#include "common/data/tuple.hpp"
#include "common/data/triple.hpp"


template<typename T>
LilMatrix<T>::LilMatrix(uint32 numRows, uint32 numCols)
    : numRows_(numRows), rows_(new Row*[numRows]) {
    for (uint32 i = 0; i < numRows; i++) {
        rows_[i] = new Row(numCols);
    }
}

template<typename T>
LilMatrix<T>::~LilMatrix() {
    for (uint32 i = 0; i < numRows_; i++) {
        Row* row = rows_[i];
        delete row;
    }

    delete[] rows_;
}

template<typename T>
typename LilMatrix<T>::iterator LilMatrix<T>::row_begin(uint32 row) {
    return rows_[row]->begin();
}

template<typename T>
typename LilMatrix<T>::iterator LilMatrix<T>::row_end(uint32 row) {
    return rows_[row]->end();
}

template<typename T>
typename LilMatrix<T>::const_iterator LilMatrix<T>::row_cbegin(uint32 row) const {
    return rows_[row]->cbegin();
}

template<typename T>
typename LilMatrix<T>::const_iterator LilMatrix<T>::row_cend(uint32 row) const {
    return rows_[row]->cend();
}

template<typename T>
typename LilMatrix<T>::Row& LilMatrix<T>::getRow(uint32 row) {
    return *rows_[row];
}

template<typename T>
const typename LilMatrix<T>::Row& LilMatrix<T>::getRow(uint32 row) const {
    return *rows_[row];
}

template<typename T>
uint32 LilMatrix<T>::getNumRows() const {
    return numRows_;
}

template<typename T>
void LilMatrix<T>::clear() {
    for (uint32 i = 0; i < numRows_; i++) {
        rows_[i]->clear();
    }
}

template class LilMatrix<uint8>;
template class LilMatrix<uint32>;
template class LilMatrix<float32>;
template class LilMatrix<float64>;
template class LilMatrix<Tuple<uint8>>;
template class LilMatrix<Tuple<uint32>>;
template class LilMatrix<Tuple<float32>>;
template class LilMatrix<Tuple<float64>>;
template class LilMatrix<Triple<uint8>>;
template class LilMatrix<Triple<uint32>>;
template class LilMatrix<Triple<float32>>;
template class LilMatrix<Triple<float64>>;
