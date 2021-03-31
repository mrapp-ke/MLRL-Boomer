#include "common/sampling/weight_vector_dense.hpp"


DenseWeightVector::DenseWeightVector(uint32 numElements)
    : vector_(DenseVector<float64>(numElements, true)), numNonZeroWeights_(0) {

}

DenseWeightVector::iterator DenseWeightVector::begin() {
    return vector_.begin();
}

DenseWeightVector::iterator DenseWeightVector::end() {
    return vector_.end();
}

DenseWeightVector::const_iterator DenseWeightVector::cbegin() const {
    return vector_.cbegin();
}

DenseWeightVector::const_iterator DenseWeightVector::cend() const {
    return vector_.cend();
}

uint32 DenseWeightVector::getNumNonZeroWeights() const {
    return numNonZeroWeights_;
}

void DenseWeightVector::setNumNonZeroWeights(uint32 numNonZeroWeights) {
    numNonZeroWeights_ = numNonZeroWeights;
}

bool DenseWeightVector::hasZeroWeights() const {
    return numNonZeroWeights_ < vector_.getNumElements();
}

float64 DenseWeightVector::getWeight(uint32 pos) const {
    return vector_.getValue(pos);
}
