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
DenseNumericVector<T>::DenseNumericVector(const DenseNumericVector<T>& vector)
    : DenseVector<T>(vector) {

}

template<class T>
void DenseNumericVector<T>::setAllToZero() {
    for (uint32 i = 0; i < DenseVector<T>::numElements_; i++) {
        DenseVector<T>::array_[i] = 0;
    }
}

template<class T>
void DenseNumericVector<T>::add(typename DenseVector<T>::const_iterator begin,
                                typename DenseVector<T>::const_iterator end) {
    for (uint32 i = 0; i < DenseVector<T>::numElements_; i++) {
        T value = begin[i];
        DenseVector<T>::array_[i] += value;
    }
}

template<class T>
void DenseNumericVector<T>::add(typename DenseVector<T>::const_iterator begin,
                                typename DenseVector<T>::const_iterator end, T weight) {
    for (uint32 i = 0; i < DenseVector<T>::numElements_; i++) {
        T value = begin[i];
        DenseVector<T>::array_[i] += (value * weight);
    }
}

template<class T>
void DenseNumericVector<T>::subtract(typename DenseVector<T>::const_iterator begin,
                                     typename DenseVector<T>::const_iterator end, T weight) {
    for (uint32 i = 0; i < DenseVector<T>::numElements_; i++) {
        T value = begin[i];
        DenseVector<T>::array_[i] -= (value * weight);
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

template<class T>
void DenseNumericVector<T>::difference(typename DenseVector<T>::const_iterator firstBegin,
                                       typename DenseVector<T>::const_iterator firstEnd,
                                       typename DenseVector<T>::const_iterator secondBegin,
                                       typename DenseVector<T>::const_iterator secondEnd) {
    for (uint32 i = 0; i < DenseVector<T>::numElements_; i++) {
        T difference = firstBegin[i] - secondBegin[i];
        DenseVector<T>::array_[i] = difference;
    }
}

template<class T>
void DenseNumericVector<T>::difference(typename DenseVector<T>::const_iterator firstBegin,
                                       typename DenseVector<T>::const_iterator firstEnd,
                                       const FullIndexVector& firstIndices,
                                       typename DenseVector<T>::const_iterator secondBegin,
                                       typename DenseVector<T>::const_iterator secondEnd) {
    for (uint32 i = 0; i < DenseVector<T>::numElements_; i++) {
        T difference = firstBegin[i] - secondBegin[i];
        DenseVector<T>::array_[i] = difference;
    }
}

template<class T>
void DenseNumericVector<T>::difference(typename DenseVector<T>::const_iterator firstBegin,
                                       typename DenseVector<T>::const_iterator firstEnd,
                                       const PartialIndexVector& firstIndices,
                                       typename DenseVector<T>::const_iterator secondBegin,
                                       typename DenseVector<T>::const_iterator secondEnd) {
    PartialIndexVector::const_iterator firstIndexIterator = firstIndices.cbegin();

    for (uint32 i = 0; i < DenseVector<T>::numElements_; i++) {
        uint32 firstIndex = firstIndexIterator[i];
        T difference = firstBegin[firstIndex] - secondBegin[i];
        DenseVector<T>::array_[i] = difference;
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
DenseNumericMatrix<T>::DenseNumericMatrix(const DenseNumericMatrix<T>& matrix)
    : DenseMatrix<T>(matrix) {

}

template<class T>
void DenseNumericMatrix<T>::setAllToZero() {
    for (uint32 i = 0; i < DenseMatrix<T>::numRows_ * DenseMatrix<T>::numCols_; i++) {
        DenseMatrix<T>::array_[i] = 0;
    }
}

template<class T>
void DenseNumericMatrix<T>::addToRow(uint32 row, typename DenseVector<T>::const_iterator begin,
                                     typename DenseVector<T>::const_iterator end) {
    uint32 offset = row * DenseMatrix<T>::numCols_;

    for (uint32 i = 0; i < DenseMatrix<T>::numCols_; i++) {
        T value = begin[i];
        DenseMatrix<T>::array_[offset + i] += value;
    }
}

template<class T>
void DenseNumericMatrix<T>::addToRowFromSubset(uint32 row, typename DenseVector<T>::const_iterator begin,
                                               typename DenseVector<T>::const_iterator end,
                                               FullIndexVector::const_iterator indicesBegin,
                                               FullIndexVector::const_iterator indicesEnd) {
    uint32 offset = row * DenseMatrix<T>::numCols_;

    for (uint32 i = 0; i < DenseMatrix<T>::numCols_; i++) {
        T value = begin[i];
        DenseMatrix<T>::array_[offset + i] += value;
    }
}

template<class T>
void DenseNumericMatrix<T>::addToRowFromSubset(uint32 row, typename DenseVector<T>::const_iterator begin,
                                               typename DenseVector<T>::const_iterator end,
                                               PartialIndexVector::const_iterator indicesBegin,
                                               PartialIndexVector::const_iterator indicesEnd) {
    uint32 offset = row * DenseMatrix<T>::numCols_;
    typename DenseVector<T>::const_iterator valueIterator = begin;

    for (auto indexIterator = indicesBegin; indexIterator != indicesEnd; indexIterator++) {
        uint32 index = *indexIterator;
        T value = *valueIterator;
        DenseMatrix<T>::array_[offset + index] += value;
        valueIterator++;
    }
}

template class DenseNumericMatrix<float64>;
