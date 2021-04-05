#include "common/data/vector_sparse_array_binary.hpp"
#include <cstdlib>


BinarySparseArrayVector::BinarySparseArrayVector(uint32 numElements)
    : array_((uint32*) malloc(numElements * sizeof(uint32))), numElements_(numElements), maxCapacity_(numElements) {

}

BinarySparseArrayVector::~BinarySparseArrayVector() {
    free(array_);
}

uint32 BinarySparseArrayVector::getNumElements() const {
    return numElements_;
}

void BinarySparseArrayVector::setNumElements(uint32 numElements, bool freeMemory) {
    if (numElements < maxCapacity_) {
        if (freeMemory) {
            array_ = (uint32*) realloc(array_, numElements * sizeof(uint32));
            maxCapacity_ = numElements;
        }
    } else if (numElements > maxCapacity_) {
        array_ = (uint32*) realloc(array_, numElements * sizeof(uint32));
        maxCapacity_ = numElements;
    }

    numElements_ = numElements;
}

BinarySparseArrayVector::index_iterator BinarySparseArrayVector::indices_begin() {
    return array_;
}

BinarySparseArrayVector::index_iterator BinarySparseArrayVector::indices_end() {
    return &array_[numElements_];
}

BinarySparseArrayVector::index_const_iterator BinarySparseArrayVector::indices_cbegin() const {
    return array_;
}

BinarySparseArrayVector::index_const_iterator BinarySparseArrayVector::indices_cend() const {
    return &array_[numElements_];
}

BinarySparseArrayVector::value_const_iterator BinarySparseArrayVector::values_cbegin() const {
    return make_index_forward_iterator(this->indices_cbegin(), this->indices_cend());
}

BinarySparseArrayVector::value_const_iterator BinarySparseArrayVector::values_cend() const {
    return make_index_forward_iterator(this->indices_cbegin(), this->indices_cend(), numElements_);
}
