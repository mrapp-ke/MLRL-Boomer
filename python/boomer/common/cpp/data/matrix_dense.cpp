#include "matrix_dense.h"
#include <cstdlib>


template<class T>
DenseMatrix<T>::DenseMatrix(uint32 numRows, uint32 numCols)
    : DenseMatrix<T>(numRows, numCols, false) {

}

template<class T>
DenseMatrix<T>::DenseMatrix(uint32 numRows, uint32 numCols, bool init)
    : array_((T*) (init ? calloc(numRows * numCols, sizeof(T)) : malloc(numRows * numCols * sizeof(T)))),
      numRows_(numRows), numCols_(numCols) {

}

template<class T>
DenseMatrix<T>::~DenseMatrix() {
    free(array_);
}

template<class T>
uint32 DenseMatrix<T>::getNumRows() const {
    return numRows_;
}

template<class T>
uint32 DenseMatrix<T>::getNumCols() const {
    return numCols_;
}

template<class T>
typename DenseMatrix<T>::iterator DenseMatrix<T>::row_begin(uint32 row) {
    return &array_[row * numCols_];
}

template<class T>
typename DenseMatrix<T>::iterator DenseMatrix<T>::row_end(uint32 row) {
    return &array_[(row + 1) * numCols_];
}

template<class T>
typename DenseMatrix<T>::const_iterator DenseMatrix<T>::row_cbegin(uint32 row) const {
    return &array_[row * numCols_];
}

template<class T>
typename DenseMatrix<T>::const_iterator DenseMatrix<T>::row_cend(uint32 row) const {
    return &array_[(row + 1) * numCols_];
}

template class DenseMatrix<float64>;
