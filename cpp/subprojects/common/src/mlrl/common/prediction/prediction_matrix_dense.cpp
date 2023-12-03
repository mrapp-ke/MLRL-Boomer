#include "mlrl/common/prediction/prediction_matrix_dense.hpp"

template<typename T>
DensePredictionMatrix<T>::DensePredictionMatrix(uint32 numRows, uint32 numCols, bool init)
    : DenseMatrixDecorator<AllocatedCContiguousView<T>>(AllocatedCContiguousView<T>(numRows, numCols, init)) {}

template<typename T>
T* DensePredictionMatrix<T>::get() {
    return this->view.array;
}

template<typename T>
T* DensePredictionMatrix<T>::release() {
    return this->view.release();
}

template class DensePredictionMatrix<uint8>;
template class DensePredictionMatrix<float64>;
