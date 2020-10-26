#include "indices.h"


DenseIndexVector::DenseIndexVector(uint32 numElements)
    : DenseVector<uint32>(numElements) {

}

RangeIndexVector::RangeIndexVector(uint32 numElements) {
    numElements_ = numElements;
}

uint32 RangeIndexVector::getNumElements() const {
    return numElements_;
}

uint32 RangeIndexVector::getValue(uint32 pos) const {
    return pos;
}
