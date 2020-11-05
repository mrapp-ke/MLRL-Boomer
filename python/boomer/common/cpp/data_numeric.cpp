#include "data_numeric.h"


template<class T>
DenseNumericVector<T>::DenseNumericVector(uint32 numElements)
    : DenseVector<T>(numElements) {

}

template<class T>
DenseNumericVector<T>::DenseNumericVector(uint32 numElements, bool init)
    : DenseVector<T>(numElements, init) {

}

template<class T>
void DenseNumericVector<T>::setAllToZero() {
    auto end = this->end();

    for (auto iterator = this->begin(); iterator != end; iterator++) {
        *iterator = 0;
    }
}

template class DenseNumericVector<float64>;
