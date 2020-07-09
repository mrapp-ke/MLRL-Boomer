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

uint8 BinaryDokMatrix::getValue(uint32 row, uint32 column) {
    return true;
}
