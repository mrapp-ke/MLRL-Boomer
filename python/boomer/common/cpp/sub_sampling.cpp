#include "sub_sampling.h"


EqualWeightVector::EqualWeightVector(uint32 numElements) {
    numElements_ = numElements;
}

uint32 EqualWeightVector::getNumElements() {
    return numElements_;
}

bool EqualWeightVector::hasZeroElements() {
    return false;
}

uint32 EqualWeightVector::getValue(uint32 pos) {
    return 1;
}

uint32 EqualWeightVector::getSumOfWeights() {
    return numElements_;
}
