#include "data.h"
#include <algorithm>
#include <cstdlib>


template<class T>
DenseVector<T>::DenseVector(uint32 numElements)
    : DenseVector<T>(numElements, false) {

}

template<class T>
DenseVector<T>::DenseVector(uint32 numElements, bool init)
    : array_((T*) (init ? calloc(numElements, sizeof(T)) : malloc(numElements * sizeof(T)))),
      numElements_(numElements) {

}

template<class T>
DenseVector<T>::~DenseVector() {
    free(array_);
}

template<class T>
uint32 DenseVector<T>::getNumElements() const {
    return numElements_;
}

template<class T>
T DenseVector<T>::getValue(uint32 pos) const {
    return array_[pos];
}

template<class T>
typename DenseVector<T>::iterator DenseVector<T>::begin() {
    return array_;
}

template<class T>
typename DenseVector<T>::iterator DenseVector<T>::end() {
    return &array_[numElements_];
}

template<class T>
typename DenseVector<T>::const_iterator DenseVector<T>::cbegin() const {
    return array_;
}

template<class T>
typename DenseVector<T>::const_iterator DenseVector<T>::cend() const {
    return &array_[numElements_];
}

template<class T>
void DenseVector<T>::setNumElements(uint32 numElements) {
    if (numElements != numElements_) {
        numElements_ = numElements;
        array_ = (T*) realloc(array_, numElements * sizeof(T));
    }
}

template class DenseVector<uint32>;
template class DenseVector<float64>;
template class DenseVector<Bin>;

template<class T>
SparseArrayVector<T>::SparseArrayVector(uint32 numElements)
    : array_((Entry*) malloc(numElements * sizeof(Entry))), numElements_(numElements) {

}

template<class T>
SparseArrayVector<T>::~SparseArrayVector() {
    free(array_);
}

template<class T>
uint32 SparseArrayVector<T>::getNumElements() const {
    return numElements_;
}

template<class T>
void SparseArrayVector<T>::setNumElements(uint32 numElements) {
    if (numElements != numElements_) {
        numElements_ = numElements;
        array_ = (Entry*) realloc(array_, numElements * sizeof(Entry));
    }
}

template<class T>
typename SparseArrayVector<T>::iterator SparseArrayVector<T>::begin() {
    return array_;
}

template<class T>
typename SparseArrayVector<T>::iterator SparseArrayVector<T>::end() {
    return &array_[numElements_];
}

template<class T>
typename SparseArrayVector<T>::const_iterator SparseArrayVector<T>::cbegin() const {
    return array_;
}

template<class T>
typename SparseArrayVector<T>::const_iterator SparseArrayVector<T>::cend() const {
    return &array_[numElements_];
}

template<class T>
void SparseArrayVector<T>::sortByValues() {
    struct {

        bool operator()(const SparseArrayVector<T>::Entry& a, const SparseArrayVector<T>::Entry& b) const {
            return a.value < b.value;
        }

    } comparator;
    std::sort(this->begin(), this->end(), comparator);
}

template class SparseArrayVector<float32>;
template class SparseArrayVector<float64>;

bool BinaryDokVector::getValue(uint32 pos) const {
    return data_.find(pos) != data_.end();
}

void BinaryDokVector::setValue(uint32 pos) {
    data_.insert(pos);
}

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
typename DenseMatrix<T>::iterator DenseMatrix<T>::begin() {
    return array_;
}

template<class T>
typename DenseMatrix<T>::iterator DenseMatrix<T>::end() {
    return &array_[numRows_ * numCols_];
}

template<class T>
typename DenseMatrix<T>::const_iterator DenseMatrix<T>::cbegin() const {
    return array_;
}

template<class T>
typename DenseMatrix<T>::const_iterator DenseMatrix<T>::cend() const {
    return &array_[numRows_ * numCols_];
}

template<class T>
typename DenseMatrix<T>::iterator DenseMatrix<T>::row_begin(uint32 row) {
    return &array_[row * numCols_];
}

template<class T>
typename DenseMatrix<T>::iterator DenseMatrix<T>::row_end(uint32 row) {
    return &array_[row * (numCols_ + 1)];
}

template<class T>
typename DenseMatrix<T>::const_iterator DenseMatrix<T>::row_cbegin(uint32 row) const {
    return &array_[row * numCols_];
}

template<class T>
typename DenseMatrix<T>::const_iterator DenseMatrix<T>::row_cend(uint32 row) const {
    return &array_[row * (numCols_ + 1)];
}

template class DenseMatrix<float64>;

bool BinaryDokMatrix::getValue(uint32 row, uint32 column) const {
    return data_.find(std::make_pair(row, column)) != data_.end();
}

void BinaryDokMatrix::setValue(uint32 row, uint32 column) {
    data_.insert(std::make_pair(row, column));
}
