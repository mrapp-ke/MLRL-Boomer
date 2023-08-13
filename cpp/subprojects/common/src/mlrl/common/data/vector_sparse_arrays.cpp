#include "mlrl/common/data/vector_sparse_arrays.hpp"

template<typename T>
SparseArraysVector<T>::SparseArraysVector(uint32 numElements)
    : indices_(DenseVector<uint32>(numElements)), values_(DenseVector<T>(numElements)) {}

template<typename T>
typename SparseArraysVector<T>::index_iterator SparseArraysVector<T>::indices_begin() {
    return indices_.begin();
}

template<typename T>
typename SparseArraysVector<T>::index_iterator SparseArraysVector<T>::indices_end() {
    return indices_.end();
}

template<typename T>
typename SparseArraysVector<T>::index_const_iterator SparseArraysVector<T>::indices_cbegin() const {
    return indices_.cbegin();
}

template<typename T>
typename SparseArraysVector<T>::index_const_iterator SparseArraysVector<T>::indices_cend() const {
    return indices_.cend();
}

template<typename T>
typename SparseArraysVector<T>::value_iterator SparseArraysVector<T>::values_begin() {
    return values_.begin();
}

template<typename T>
typename SparseArraysVector<T>::value_iterator SparseArraysVector<T>::values_end() {
    return values_.end();
}

template<typename T>
typename SparseArraysVector<T>::value_const_iterator SparseArraysVector<T>::values_cbegin() const {
    return values_.cbegin();
}

template<typename T>
typename SparseArraysVector<T>::value_const_iterator SparseArraysVector<T>::values_cend() const {
    return values_.cend();
}

template<typename T>
uint32 SparseArraysVector<T>::getNumElements() const {
    return indices_.getNumElements();
}

template class SparseArraysVector<uint8>;
template class SparseArraysVector<uint32>;
template class SparseArraysVector<float32>;
template class SparseArraysVector<float64>;
