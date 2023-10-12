#include "mlrl/common/data/vector_dense.hpp"

#include "mlrl/common/data/indexed_value.hpp"
#include "mlrl/common/data/triple.hpp"
#include "mlrl/common/data/tuple.hpp"

#include <cstdlib>

template<typename T>
DenseVector<T>::DenseVector(uint32 numElements) : DenseVector<T>(numElements, false) {}

template<typename T>
DenseVector<T>::DenseVector(uint32 numElements, bool init)
    : VectorView<T>(numElements, (T*) (init ? calloc(numElements, sizeof(T)) : malloc(numElements * sizeof(T)))),
      maxCapacity_(numElements) {}

template<typename T>
DenseVector<T>::~DenseVector() {
    free(this->array_);
}

template<typename T>
void DenseVector<T>::setNumElements(uint32 numElements, bool freeMemory) {
    if (numElements < maxCapacity_) {
        if (freeMemory) {
            this->array_ = (T*) realloc(this->array_, numElements * sizeof(T));
            maxCapacity_ = numElements;
        }
    } else if (numElements > maxCapacity_) {
        this->array_ = (T*) realloc(this->array_, numElements * sizeof(T));
        maxCapacity_ = numElements;
    }

    this->numElements_ = numElements;
}

template class DenseVector<uint8>;
template class DenseVector<uint32>;
template class DenseVector<int32>;
template class DenseVector<int64>;
template class DenseVector<float32>;
template class DenseVector<float64>;
template class DenseVector<IndexedValue<uint8>>;
template class DenseVector<IndexedValue<uint32>>;
template class DenseVector<IndexedValue<int32>>;
template class DenseVector<IndexedValue<int64>>;
template class DenseVector<IndexedValue<float32>>;
template class DenseVector<IndexedValue<float64>>;
template class DenseVector<Tuple<uint8>>;
template class DenseVector<Tuple<uint32>>;
template class DenseVector<Tuple<int32>>;
template class DenseVector<Tuple<int64>>;
template class DenseVector<Tuple<float32>>;
template class DenseVector<Tuple<float64>>;
template class DenseVector<IndexedValue<Tuple<uint8>>>;
template class DenseVector<IndexedValue<Tuple<uint32>>>;
template class DenseVector<IndexedValue<Tuple<int32>>>;
template class DenseVector<IndexedValue<Tuple<int64>>>;
template class DenseVector<IndexedValue<Tuple<float32>>>;
template class DenseVector<IndexedValue<Tuple<float64>>>;
template class DenseVector<Triple<uint8>>;
template class DenseVector<Triple<uint32>>;
template class DenseVector<Triple<int32>>;
template class DenseVector<Triple<int64>>;
template class DenseVector<Triple<float32>>;
template class DenseVector<Triple<float64>>;
template class DenseVector<IndexedValue<Triple<uint8>>>;
template class DenseVector<IndexedValue<Triple<uint32>>>;
template class DenseVector<IndexedValue<Triple<int32>>>;
template class DenseVector<IndexedValue<Triple<int64>>>;
template class DenseVector<IndexedValue<Triple<float32>>>;
template class DenseVector<IndexedValue<Triple<float64>>>;
