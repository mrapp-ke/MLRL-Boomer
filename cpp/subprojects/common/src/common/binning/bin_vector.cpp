#include "common/binning/bin_vector.hpp"


BinVectorNew::BinVectorNew(uint32 numElements)
    : BinVectorNew(numElements, false) {

}

BinVectorNew::BinVectorNew(uint32 numElements, bool init)
    : DenseVector<Bin>(numElements) {
    if (init) {
        DenseVector<Bin>::iterator iterator = this->begin();

        for (uint32 i = 0; i < numElements; i++) {
            new (iterator + i) Bin();
            iterator[i].index = i;
        }
    }
}

BinVector::BinVector(uint32 numElements)
    : BinVector(numElements, false) {

}

BinVector::BinVector(uint32 numElements, bool init)
    : vector_(DenseVector<Bin>(numElements)), mapping_(DenseMappingVector<Example>(numElements)) {
    if (init) {
        DenseVector<Bin>::iterator iterator = vector_.begin();

        for (uint32 i = 0; i < numElements; i++) {
            new (iterator + i) Bin();
            iterator[i].index = i;
        }
    }
}

BinVector::bin_iterator BinVector::bins_begin() {
    return vector_.begin();
}

BinVector::bin_iterator BinVector::bins_end() {
    return vector_.end();
}

BinVector::bin_const_iterator BinVector::bins_cbegin() const {
    return vector_.cbegin();
}

BinVector::bin_const_iterator BinVector::bins_cend() const {
    return vector_.cend();
}

BinVector::example_list_iterator BinVector::examples_begin() {
    return mapping_.begin();
}

BinVector::example_list_iterator BinVector::examples_end() {
    return mapping_.end();
}

BinVector::example_list_const_iterator BinVector::examples_cbegin() const {
    return mapping_.cbegin();
}

BinVector::example_list_const_iterator BinVector::examples_cend() const {
    return mapping_.cend();
}

uint32 BinVector::getNumElements() const {
    return vector_.getNumElements();
}

void BinVector::setNumElements(uint32 numElements, bool freeMemory) {
    vector_.setNumElements(numElements, freeMemory);
    mapping_.setNumElements(numElements, freeMemory);
}

void BinVector::swapExamples(uint32 pos1, uint32 pos2) {
    mapping_.swap(pos1, pos2);
}
