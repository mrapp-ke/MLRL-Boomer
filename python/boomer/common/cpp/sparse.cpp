#include "sparse.h"


BinaryDokMatrix::BinaryDokMatrix(uint32 numRows, uint32 numCols) {
    numRows_ = numRows;
    numCols_ = numCols;
}

uint32 BinaryDokMatrix::getNumRows() {
    return numRows_;
}

uint32 BinaryDokMatrix::getNumCols() {
    return numCols_;
}

void BinaryDokMatrix::set(uint32 row, uint32 column) {
    data_.insert(std::make_pair(row, column));
}

uint8 BinaryDokMatrix::get(uint32 row, uint32 column) {
    return data_.find(std::make_pair(row, column)) != data_.end();
}

BinaryDokVector::BinaryDokVector(uint32 numElements) {
    numElements_ = numElements;
}

uint32 BinaryDokVector::getNumElements() {
    return numElements_;
}

void BinaryDokVector::set(uint32 pos) {
    data_.insert(pos);
}

uint8 BinaryDokVector::get(uint32 pos) {
    return data_.find(pos) != data_.end();
}
