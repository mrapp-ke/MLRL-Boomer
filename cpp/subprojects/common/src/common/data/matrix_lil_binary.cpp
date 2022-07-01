#include "common/data/matrix_lil_binary.hpp"


BinaryLilMatrix::BinaryLilMatrix(uint32 numRows)
    : numRows_(numRows), array_(new std::vector<uint32>[numRows] {}) {

}

BinaryLilMatrix::~BinaryLilMatrix() {
    delete[] array_;
}

BinaryLilMatrix::iterator BinaryLilMatrix::row_begin(uint32 row) {
    return array_[row].begin();
}

BinaryLilMatrix::iterator BinaryLilMatrix::row_end(uint32 row) {
    return array_[row].end();
}

BinaryLilMatrix::const_iterator BinaryLilMatrix::row_cbegin(uint32 row) const {
    return array_[row].cbegin();
}

BinaryLilMatrix::const_iterator BinaryLilMatrix::row_cend(uint32 row) const {
    return array_[row].cend();
}

BinaryLilMatrix::row BinaryLilMatrix::operator[](uint32 row) {
    return array_[row];
}

BinaryLilMatrix::const_row BinaryLilMatrix::operator[](uint32 row) const {
    return array_[row];
}

uint32 BinaryLilMatrix::getNumRows() const {
    return numRows_;
}

void BinaryLilMatrix::clear() {
    for (uint32 i = 0; i < numRows_; i++) {
        array_[i].clear();
    }
}
