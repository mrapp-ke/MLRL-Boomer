#include "mlrl/common/data/view_vector.hpp"

#include "mlrl/common/data/indexed_value.hpp"
#include "mlrl/common/data/triple.hpp"
#include "mlrl/common/data/tuple.hpp"

template<typename T>
VectorConstView<T>::VectorConstView(uint32 numElements, T* array) : numElements_(numElements), array_(array) {}

template<typename T>
typename VectorConstView<T>::const_iterator VectorConstView<T>::cbegin() const {
    return array_;
}

template<typename T>
typename VectorConstView<T>::const_iterator VectorConstView<T>::cend() const {
    return &array_[numElements_];
}

template<typename T>
const T& VectorConstView<T>::operator[](uint32 pos) const {
    return array_[pos];
}

template<typename T>
uint32 VectorConstView<T>::getNumElements() const {
    return numElements_;
}

template class VectorConstView<uint8>;
template class VectorConstView<const uint8>;
template class VectorConstView<uint32>;
template class VectorConstView<const uint32>;
template class VectorConstView<int64>;
template class VectorConstView<const int64>;
template class VectorConstView<float32>;
template class VectorConstView<const float32>;
template class VectorConstView<float64>;
template class VectorConstView<const float64>;
template class VectorConstView<IndexedValue<uint8>>;
template class VectorConstView<IndexedValue<uint32>>;
template class VectorConstView<IndexedValue<int64>>;
template class VectorConstView<IndexedValue<float32>>;
template class VectorConstView<IndexedValue<float64>>;
template class VectorConstView<Tuple<uint8>>;
template class VectorConstView<Tuple<uint32>>;
template class VectorConstView<Tuple<int64>>;
template class VectorConstView<Tuple<float32>>;
template class VectorConstView<Tuple<float64>>;
template class VectorConstView<IndexedValue<Tuple<uint8>>>;
template class VectorConstView<IndexedValue<Tuple<uint32>>>;
template class VectorConstView<IndexedValue<Tuple<int64>>>;
template class VectorConstView<IndexedValue<Tuple<float32>>>;
template class VectorConstView<IndexedValue<Tuple<float64>>>;
template class VectorConstView<Triple<uint8>>;
template class VectorConstView<Triple<uint32>>;
template class VectorConstView<Triple<int64>>;
template class VectorConstView<Triple<float32>>;
template class VectorConstView<Triple<float64>>;
template class VectorConstView<IndexedValue<Triple<uint8>>>;
template class VectorConstView<IndexedValue<Triple<uint32>>>;
template class VectorConstView<IndexedValue<Triple<int64>>>;
template class VectorConstView<IndexedValue<Triple<float32>>>;
template class VectorConstView<IndexedValue<Triple<float64>>>;
