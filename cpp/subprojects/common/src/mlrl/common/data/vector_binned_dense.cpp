#include "mlrl/common/data/vector_binned_dense.hpp"

template<typename T>
DenseBinnedVector<T>::DenseBinnedVector(uint32 numElements, uint32 numBins)
    : binIndices_(numElements), values_(numBins) {}

template<typename T>
typename DenseBinnedVector<T>::const_iterator DenseBinnedVector<T>::cbegin() const {
    return BinnedConstIterator<T>(binIndices_.cbegin(), values_.cbegin());
}

template<typename T>
typename DenseBinnedVector<T>::const_iterator DenseBinnedVector<T>::cend() const {
    return BinnedConstIterator<T>(binIndices_.cend(), values_.cbegin());
}

template<typename T>
typename DenseBinnedVector<T>::index_iterator DenseBinnedVector<T>::indices_begin() {
    return binIndices_.begin();
}

template<typename T>
typename DenseBinnedVector<T>::index_iterator DenseBinnedVector<T>::indices_end() {
    return binIndices_.end();
}

template<typename T>
typename DenseBinnedVector<T>::index_const_iterator DenseBinnedVector<T>::indices_cbegin() const {
    return binIndices_.cbegin();
}

template<typename T>
typename DenseBinnedVector<T>::index_const_iterator DenseBinnedVector<T>::indices_cend() const {
    return binIndices_.cend();
}

template<typename T>
typename DenseBinnedVector<T>::value_iterator DenseBinnedVector<T>::values_begin() {
    return values_.begin();
}

template<typename T>
typename DenseBinnedVector<T>::value_iterator DenseBinnedVector<T>::values_end() {
    return values_.end();
}

template<typename T>
typename DenseBinnedVector<T>::value_const_iterator DenseBinnedVector<T>::values_cbegin() const {
    return values_.cbegin();
}

template<typename T>
typename DenseBinnedVector<T>::value_const_iterator DenseBinnedVector<T>::values_cend() const {
    return values_.cend();
}

template<typename T>
uint32 DenseBinnedVector<T>::getNumElements() const {
    return binIndices_.getNumElements();
}

template<typename T>
uint32 DenseBinnedVector<T>::getNumBins() const {
    return values_.getNumElements();
}

template<typename T>
void DenseBinnedVector<T>::setNumBins(uint32 numBins, bool freeMemory) {
    values_.setNumElements(numBins, freeMemory);
}

template class DenseBinnedVector<uint8>;
template class DenseBinnedVector<uint32>;
template class DenseBinnedVector<int64>;
template class DenseBinnedVector<float32>;
template class DenseBinnedVector<float64>;
