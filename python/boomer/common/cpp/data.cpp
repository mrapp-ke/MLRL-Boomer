#include "data.h"


template<class T>
DenseVector<T>::DenseVector(uint32 numElements)
    : DenseVector<T>(numElements, false) {

}

template<class T>
DenseVector<T>::DenseVector(uint32 numElements, bool init)
    : array_(init ? new T[numElements]() : new T[numElements]), numElements_(numElements) {

}

template<class T>
DenseVector<T>::~DenseVector() {
    delete[] array_;
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
    return &array_[0];
}

template<class T>
typename DenseVector<T>::iterator DenseVector<T>::end() {
    return &array_[numElements_];
}

template<class T>
typename DenseVector<T>::const_iterator DenseVector<T>::cbegin() const {
    return &array_[0];
}

template<class T>
typename DenseVector<T>::const_iterator DenseVector<T>::cend() const {
    return &array_[numElements_];
}

template class DenseVector<uint32>;

DenseIndexVector::DenseIndexVector(const uint32* indices, uint32 numElements)
    : indices_(indices), numElements_(numElements) {

}

DenseIndexVector::~DenseIndexVector() {
    delete[] indices_;
}

uint32 DenseIndexVector::getNumElements() const {
    return numElements_;
}

uint32 DenseIndexVector::getIndex(uint32 pos) const {
    return indices_[pos];
}

RangeIndexVector::RangeIndexVector(uint32 numIndices) {
    numIndices_ = numIndices;
}

uint32 RangeIndexVector::getNumElements() const {
    return numIndices_;
}

uint32 RangeIndexVector::getIndex(uint32 pos) const {
    return pos;
}

BinaryDokVector::BinaryDokVector(uint32 numElements)
    : numElements_(numElements) {

}

uint32 BinaryDokVector::getNumElements() const {
    return numElements_;
}

uint8 BinaryDokVector::getValue(uint32 pos) const {
    return data_.find(pos) != data_.end();
}

void BinaryDokVector::setValue(uint32 pos) {
    data_.insert(pos);
}

BinaryDokMatrix::BinaryDokMatrix(uint32 numRows, uint32 numCols)
    : numRows_(numRows), numCols_(numCols) {

}

uint32 BinaryDokMatrix::getNumRows() const {
    return numRows_;
}

uint32 BinaryDokMatrix::getNumCols() const {
    return numCols_;
}

uint8 BinaryDokMatrix::getValue(uint32 row, uint32 column) const {
    return data_.find(std::make_pair(row, column)) != data_.end();
}

void BinaryDokMatrix::setValue(uint32 row, uint32 column) {
    data_.insert(std::make_pair(row, column));
}
