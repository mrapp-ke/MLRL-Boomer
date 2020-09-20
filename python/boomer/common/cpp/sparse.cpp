#include "sparse.h"


void BinaryDokMatrix::addValue(uint32 row, uint32 column) {
    data_.insert(std::make_pair(row, column));
}

uint8 BinaryDokMatrix::getValue(uint32 row, uint32 column) {
    return data_.find(std::make_pair(row, column)) != data_.end();
}

void BinaryDokVector::addValue(uint32 pos) {
    data_.insert(pos);
}

uint8 BinaryDokVector::getValue(uint32 pos) {
    return data_.find(pos) != data_.end();
}
