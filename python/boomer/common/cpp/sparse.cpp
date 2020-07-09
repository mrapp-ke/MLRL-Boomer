#include "sparse.h"

using namespace sparse;


BinaryDokMatrix::BinaryDokMatrix(uint32 width, uint32 height) {
    width_ = width;
    height_ = height;
}

uint32 BinaryDokMatrix::getWidth() {
    return width_;
}

uint32 BinaryDokMatrix::getHeight() {
    return height_;
}

void BinaryDokMatrix::addValue(uint32 row, uint32 column) {
    data_.insert(std::make_pair(row, column));
}

uint8 BinaryDokMatrix::getValue(uint32 row, uint32 column) {
    return data_.find(std::make_pair(row, column)) != data_.end();
}
