#include "common/sampling/weight_vector_out_of_sample.hpp"
#include "common/sampling/weight_vector_equal.hpp"
#include "common/sampling/weight_vector_bit.hpp"
#include "common/sampling/weight_vector_dense.hpp"


template<typename T>
OutOfSampleWeightVector<T>::OutOfSampleWeightVector(const T& vector)
    : vector_(vector) {

}

template<typename T>
uint32 OutOfSampleWeightVector<T>::getNumElements() const {
    return vector_.getNumElements();
}

template<typename T>
bool OutOfSampleWeightVector<T>::operator[](uint32 pos) const {
    return vector_[pos] == 0;
}

template<typename T>
float64 OutOfSampleWeightVector<T>::getWeight(uint32 pos) const {
    return vector_.getWeight(pos) > 0 ? 0 : 1;
}

template class OutOfSampleWeightVector<EqualWeightVector>;
template class OutOfSampleWeightVector<BitWeightVector>;
template class OutOfSampleWeightVector<DenseWeightVector<uint32>>;
