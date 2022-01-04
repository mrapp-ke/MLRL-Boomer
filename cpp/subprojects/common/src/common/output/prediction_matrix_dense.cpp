#include "common/output/prediction_matrix_dense.hpp"
#include <cstdlib>


template<typename T>
DensePredictionMatrix<T>::DensePredictionMatrix(uint32 numRows, uint32 numCols, T* array)
    : CContiguousConstView<T>(numRows, numCols, array), array_(array) {

}

template<typename T>
DensePredictionMatrix<T>::~DensePredictionMatrix() {
    free(array_);
}

template<typename T>
T* DensePredictionMatrix<T>::release() {
    T* ptr = array_;
    array_ = nullptr;
    return ptr;
}

template class DensePredictionMatrix<uint8>;
template class DensePredictionMatrix<uint32>;
template class DensePredictionMatrix<float32>;
template class DensePredictionMatrix<float64>;
