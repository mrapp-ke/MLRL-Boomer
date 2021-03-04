#include "common/binning/bin_index_vector_dense.hpp"


DenseBinIndexVector::DenseBinIndexVector(uint32 numElements)
    : DenseBinIndexVector(numElements, false) {

}

DenseBinIndexVector::DenseBinIndexVector(uint32 numElements, bool init)
    : vector_(DenseVector<uint32>(numElements, init)) {

}

DenseBinIndexVector::iterator DenseBinIndexVector::begin() {
    return vector_.begin();
}

DenseBinIndexVector::iterator DenseBinIndexVector::end() {
    return vector_.end();
}

DenseBinIndexVector::const_iterator DenseBinIndexVector::cbegin() const {
    return vector_.cbegin();
}

DenseBinIndexVector::const_iterator DenseBinIndexVector::cend() const {
    return vector_.cend();
}

uint32 DenseBinIndexVector::getNumElements() const {
    return vector_.getNumElements();
}

void DenseBinIndexVector::setNumElements(uint32 numElements, bool freeMemory) {
    vector_.setNumElements(numElements, freeMemory);
}

uint32 DenseBinIndexVector::getBinIndex(uint32 exampleIndex) const {
    return vector_.getValue(exampleIndex);
}
