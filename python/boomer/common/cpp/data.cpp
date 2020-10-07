#include "data.h"


RangeIndexVector::RangeIndexVector(uint32 numIndices) {
    numIndices_ = numIndices;
}

uint32 RangeIndexVector::getNumElements() const {
    return numIndices_;
}

uint32 RangeIndexVector::getIndex(uint32 pos) const {
    return pos;
}

BinaryDokVector::BinaryDokVector(uint32 numElements) {
    numElements_ = numElements;
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

BinaryDokMatrix::BinaryDokMatrix(uint32 numRows, uint32 numCols) {
    numRows_ = numRows;
    numCols_ = numCols;
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
