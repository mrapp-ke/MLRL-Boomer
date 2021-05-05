#include "common/data/vector_sparse_array_binary.hpp"


BinarySparseArrayVector::BinarySparseArrayVector(uint32 numElements)
    : BinarySparseArrayVector(numElements, false) {

}

BinarySparseArrayVector::BinarySparseArrayVector(uint32 numElements, bool init)
    : vector_(DenseVector<uint32>(numElements, init)) {

}

BinarySparseArrayVector::index_iterator BinarySparseArrayVector::indices_begin() {
    return vector_.begin();
}

BinarySparseArrayVector::index_iterator BinarySparseArrayVector::indices_end() {
    return vector_.end();
}

BinarySparseArrayVector::index_const_iterator BinarySparseArrayVector::indices_cbegin() const {
    return vector_.cbegin();
}

BinarySparseArrayVector::index_const_iterator BinarySparseArrayVector::indices_cend() const {
    return vector_.cend();
}

BinarySparseArrayVector::value_const_iterator BinarySparseArrayVector::values_cbegin() const {
    return make_index_forward_iterator(vector_.cbegin(), vector_.cend());
}

BinarySparseArrayVector::value_const_iterator BinarySparseArrayVector::values_cend() const {
    return make_index_forward_iterator(vector_.cbegin(), vector_.cend(), vector_.getNumElements());
}

uint32 BinarySparseArrayVector::getNumElements() const {
    return vector_.getNumElements();
}

void BinarySparseArrayVector::setNumElements(uint32 numElements, bool freeMemory) {
    vector_.setNumElements(numElements, freeMemory);
}
