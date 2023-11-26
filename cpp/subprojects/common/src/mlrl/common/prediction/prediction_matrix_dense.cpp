#include "mlrl/common/prediction/prediction_matrix_dense.hpp"

#include "mlrl/common/util/memory.hpp"

template<typename T>
DensePredictionMatrix<T>::DensePredictionMatrix(uint32 numRows, uint32 numCols, bool init)
    : CContiguousView<T>(allocateMemory<T>(numRows * numCols, init), numRows, numCols),
      array_(CContiguousView<T>::array) {}

template<typename T>
DensePredictionMatrix<T>::~DensePredictionMatrix() {
    freeMemory(array_);
}

template<typename T>
T* DensePredictionMatrix<T>::get() {
    return array_;
}

template<typename T>
T* DensePredictionMatrix<T>::release() {
    T* ptr = array_;
    array_ = nullptr;
    return ptr;
}

template class DensePredictionMatrix<uint8>;
template class DensePredictionMatrix<float64>;
