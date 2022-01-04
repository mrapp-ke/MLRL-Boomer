#include "common/output/prediction_matrix_dense.hpp"
#include <cstdlib>


template<typename T>
DensePredictionMatrix<T>::DensePredictionMatrix(uint32 numRows, uint32 numCols)
    : CContiguousConstView<T>(numRows, numCols, (T*) malloc(numRows * numCols * sizeof(T))), matrix_(this->array_) {

}

template<typename T>
DensePredictionMatrix<T>::~DensePredictionMatrix() {
    free(matrix_);
}

template<typename T>
T* DensePredictionMatrix<T>::release() {
    T* ptr = matrix_;
    matrix_ = nullptr;
    return ptr;
}

template class DensePredictionMatrix<uint8>;
template class DensePredictionMatrix<uint32>;
template class DensePredictionMatrix<float32>;
template class DensePredictionMatrix<float64>;
