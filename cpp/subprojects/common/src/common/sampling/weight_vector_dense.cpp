#include "common/sampling/weight_vector_dense.hpp"


template<class T>
DenseWeightVector<T>::DenseWeightVector(uint32 numElements)
    : vector_(DenseVector<T>(numElements, true)), numNonZeroWeights_(0) {

}

template<class T>
typename DenseWeightVector<T>::iterator DenseWeightVector<T>::begin() {
    return vector_.begin();
}

template<class T>
typename DenseWeightVector<T>::iterator DenseWeightVector<T>::end() {
    return vector_.end();
}

template<class T>
typename DenseWeightVector<T>::const_iterator DenseWeightVector<T>::cbegin() const {
    return vector_.cbegin();
}

template<class T>
typename DenseWeightVector<T>::const_iterator DenseWeightVector<T>::cend() const {
    return vector_.cend();
}

template<class T>
uint32 DenseWeightVector<T>::getNumNonZeroWeights() const {
    return numNonZeroWeights_;
}

template<class T>
void DenseWeightVector<T>::setNumNonZeroWeights(uint32 numNonZeroWeights) {
    numNonZeroWeights_ = numNonZeroWeights;
}

template<class T>
bool DenseWeightVector<T>::hasZeroWeights() const {
    return numNonZeroWeights_ < vector_.getNumElements();
}

template<class T>
float64 DenseWeightVector<T>::getWeight(uint32 pos) const {
    return (float64) vector_.getValue(pos);
}

template class DenseWeightVector<uint32>;
