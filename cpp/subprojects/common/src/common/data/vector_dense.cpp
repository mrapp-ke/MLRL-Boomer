#include "common/data/vector_dense.hpp"
#include <cstdlib>


template<class T>
DenseVector<T>::DenseVector(uint32 numElements)
    : DenseVector<T>(numElements, false) {

}

template<class T>
DenseVector<T>::DenseVector(uint32 numElements, bool init)
    : VectorView<T>(numElements, (T*) (init ? calloc(numElements, sizeof(T)) : malloc(numElements * sizeof(T)))),
      maxCapacity_(numElements) {

}

template<class T>
DenseVector<T>::~DenseVector() {
    free(VectorView<T>::array_);
}

template<class T>
T DenseVector<T>::getValue(uint32 pos) const {
    return VectorView<T>::array_[pos];
}

template<class T>
void DenseVector<T>::setValue(uint32 pos, T value) {
    VectorView<T>::array_[pos] = value;
}

template<class T>
void DenseVector<T>::setNumElements(uint32 numElements, bool freeMemory) {
    if (numElements < maxCapacity_) {
        if (freeMemory) {
            DenseVector<T>::array_ = (T*) realloc(DenseVector<T>::array_, numElements * sizeof(T));
            maxCapacity_ = numElements;
        }
    } else if (numElements > maxCapacity_) {
        DenseVector<T>::array_ = (T*) realloc(DenseVector<T>::array_, numElements * sizeof(T));
        maxCapacity_ = numElements;
    }

    DenseVector<T>::numElements_ = numElements;
}

template class DenseVector<uint8>;
template class DenseVector<uint32>;
template class DenseVector<float32>;
template class DenseVector<float64>;
