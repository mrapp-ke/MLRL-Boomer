#include "mlrl/common/data/matrix_c_contiguous.hpp"

#include "mlrl/common/util/memory.hpp"

template<typename T>
CContiguousMatrix<T>::CContiguousMatrix(uint32 numRows, uint32 numCols, bool init)
    : CContiguousView<T>(allocateMemory<T>(numRows * numCols, init), numRows, numCols) {}

template<typename T>
CContiguousMatrix<T>::~CContiguousMatrix() {
    freeMemory(this->array);
}

template class CContiguousMatrix<uint8>;
template class CContiguousMatrix<uint32>;
template class CContiguousMatrix<int64>;
template class CContiguousMatrix<float32>;
template class CContiguousMatrix<float64>;
