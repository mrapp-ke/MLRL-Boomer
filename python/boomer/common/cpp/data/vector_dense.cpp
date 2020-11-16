#include "vector_dense.h"
#include "bin.h"
#include <cstdlib>


template<class T>
DenseVector<T>::DenseVector(uint32 numElements)
    : DenseVector<T>(numElements, false) {

}

template<class T>
DenseVector<T>::DenseVector(uint32 numElements, bool init)
    : array_((T*) (init ? calloc(numElements, sizeof(T)) : malloc(numElements * sizeof(T)))),
      numElements_(numElements) {

}

template<class T>
DenseVector<T>::~DenseVector() {
    free(array_);
}

template<class T>
uint32 DenseVector<T>::getNumElements() const {
    return numElements_;
}

template<class T>
T DenseVector<T>::getValue(uint32 pos) const {
    return array_[pos];
}

template<class T>
typename DenseVector<T>::iterator DenseVector<T>::begin() {
    return array_;
}

template<class T>
typename DenseVector<T>::iterator DenseVector<T>::end() {
    return &array_[numElements_];
}

template<class T>
typename DenseVector<T>::const_iterator DenseVector<T>::cbegin() const {
    return array_;
}

template<class T>
typename DenseVector<T>::const_iterator DenseVector<T>::cend() const {
    return &array_[numElements_];
}

template<class T>
void DenseVector<T>::setNumElements(uint32 numElements) {
    if (numElements != numElements_) {
        numElements_ = numElements;
        array_ = (T*) realloc(array_, numElements * sizeof(T));
    }
}

template class DenseVector<uint32>;
template class DenseVector<float64>;
template class DenseVector<Bin>;
