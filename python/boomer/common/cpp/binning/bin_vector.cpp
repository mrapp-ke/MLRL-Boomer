#include "bin_vector.h"


BinVector::BinVector(uint32 numElements)
    : BinVector(numElements, false) {

}

BinVector::BinVector(uint32 numElements, bool init)
    : vector_(DenseVector<Bin>(numElements)) {
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

BinVector::ExampleList& BinVector::getExamples(uint32 binIndex) {
    return examplesPerBin_[binIndex];
}

uint32 BinVector::getNumElements() const {
    return vector_.getNumElements();
}

void BinVector::setNumElements(uint32 numElements, bool freeMemory) {
    vector_.setNumElements(numElements, freeMemory);
}
