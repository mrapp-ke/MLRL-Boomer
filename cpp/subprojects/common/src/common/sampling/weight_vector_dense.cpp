#include "common/sampling/weight_vector_dense.hpp"


DenseWeightVector::DenseWeightVector(uint32 numElements, float64 sumOfWeights)
    : vector_(DenseVector<float64>(numElements, true)), sumOfWeights_(sumOfWeights) {

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

bool DenseWeightVector::hasZeroWeights() const {
    return true;
}

float64 DenseWeightVector::getWeight(uint32 pos) const {
    return vector_.getValue(pos);
}

float64 DenseWeightVector::getSumOfWeights() const {
    return sumOfWeights_;
}
