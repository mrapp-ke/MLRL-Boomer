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
uint32 VectorView<T>::getNumElements() const {
    return numElements_;
}
