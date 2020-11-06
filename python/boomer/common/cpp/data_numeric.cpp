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
    for (uint32 i = 0; i < DenseVector<T>::numElements_; i++) {
        DenseVector<T>::array_[i] = 0;
    }
}

template<class T>
void DenseNumericVector<T>::addToSubset(typename DenseVector<T>::const_iterator begin,
                                        typename DenseVector<T>::const_iterator end,
                                        const FullIndexVector& indices, T weight) {
    for (uint32 i = 0; i < DenseVector<T>::numElements_; i++) {
        T value = begin[i];
        DenseVector<T>::array_[i] += (value * weight);
    }
}

template<class T>
void DenseNumericVector<T>::addToSubset(typename DenseVector<T>::const_iterator begin,
                                        typename DenseVector<T>::const_iterator end,
                                        const PartialIndexVector& indices, T weight) {
    PartialIndexVector::const_iterator indexIterator = indices.cbegin();

    for (uint32 i = 0; i < DenseVector<T>::numElements_; i++) {
        uint32 index = indexIterator[i];
        T value = begin[index];
        DenseVector<T>::array_[i] += (value * weight);
    }
}

template class DenseNumericVector<float64>;

template<class T>
DenseNumericMatrix<T>::DenseNumericMatrix(uint32 numRows, uint32 numCols)
    : DenseMatrix<T>(numRows, numCols) {

}

template<class T>
DenseNumericMatrix<T>::DenseNumericMatrix(uint32 numRows, uint32 numCols, bool init)
    : DenseMatrix<T>(numRows, numCols, init) {

}

template<class T>
void DenseNumericMatrix<T>::setAllToZero() {
    for (uint32 i = 0; i < DenseMatrix<T>::numRows_ * DenseMatrix<T>::numCols_; i++) {
        DenseMatrix<T>::array_[i] = 0;
    }
}

template class DenseNumericMatrix<float64>;
