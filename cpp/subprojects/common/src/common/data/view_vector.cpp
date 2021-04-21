#include "common/data/view_vector.hpp"


template<class T>
VectorView<T>::VectorView(uint32 numElements, T* array)
    : numElements_(numElements), array_(array) {

}

template<class T>
typename VectorView<T>::iterator VectorView<T>::begin() {
    return array_;
}

template<class T>
typename VectorView<T>::iterator VectorView<T>::end() {
    return &array_[numElements_];
}

template<class T>
typename VectorView<T>::const_iterator VectorView<T>::cbegin() const {
    return array_;
}

template<class T>
typename VectorView<T>::const_iterator VectorView<T>::cend() const {
    return &array_[numElements_];
}

template<class T>
T VectorView<T>::getValue(uint32 pos) const {
    return array_[pos];
}

template<class T>
void VectorView<T>::setValue(uint32 pos, T value) {
    array_[pos] = value;
}

template<class T>
uint32 VectorView<T>::getNumElements() const {
    return numElements_;
}

template class VectorView<uint8>;
template class VectorView<uint32>;
template class VectorView<float32>;
template class VectorView<float64>;
